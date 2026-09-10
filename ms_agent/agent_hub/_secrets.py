# Copyright (c) ModelScope Contributors. All rights reserved.
"""Content-driven outbound secret redaction (BUG-0909-01).

Why this module exists
----------------------
:meth:`WorkspaceSpec.sanitize_outbound_file` decides *whether* to clean a file
by its PATH: each framework whitelists its own root config (ms-agent
``settings.json`` / ``mcp.json``, qwenpaw ``agent.json``, hermes
``config.yaml``, openhuman ``config.toml``) and returns every other collected
file verbatim -- openclaw, nanobot and qoder define no hook at all. The
collect patterns, however, take ``skills/*`` recursively plus the persona and
memory documents, so a key that an AI assistant wrote into a skill script, a
skill-local ``mcp.json``, ``SOUL.md`` or ``MEMORY.md`` was uploaded verbatim
into the remote repo and its git history.

This layer decides on CONTENT instead of path. It runs *after* the
per-framework hook (see :func:`ms_agent.agent_hub._sync.sanitize_outbound`),
so the structural cleaning and the fail-closed refusals for the known config
files keep their behavior, and every framework -- present or future -- is
covered without editing a single spec.

Two tiers
---------
* **Tier A (config-shaped)** -- a file that IS configuration, by name
  (``mcp.json``, ``config.yaml``, ...) or by shape (its parsed JSON/YAML/TOML
  carries an ``mcpServers`` / ``mcp_servers`` mapping), is cleaned with the
  structural scrubbers in :mod:`._workspace`. A shape-triggered file is
  cleaned only inside that MCP subtree: the shared vocabulary blanks any key
  named ``tokens`` / ``keys`` / ``session_id``, which would destroy legitimate
  data in a skill's JSON fixture or a memory dump.
* **Tier B (everything else)** -- documents, scripts, notebooks, JSONL: only
  high-confidence secret *values* are replaced with a ``[REDACTED:<kind>]``
  marker. The bag rules (``env`` / ``headers`` cleared wholesale) never apply
  here, and the name vocabulary is narrower than :func:`is_secret_key`
  (``max_tokens`` / ``page_token`` / ``session_id`` are pagination and
  telemetry fields, not credentials).

Contract
--------
* **Never raises.** The watch daemon swallows exceptions and keeps polling
  (``_watcher._poll_once``), so a raise here would silently stop all syncing.
* **Returns the original bytes object when nothing was redacted.**
  Re-serializing a parsed config reformats it, which breaks both
  :func:`._sync.drop_unchanged_defaults` (byte comparison against the
  framework default templates) and the sha256 idempotent-skip in
  ``push_mirror`` / ``push_resources``.
* **Deterministic and idempotent.** The watcher stores the sha256 of the
  SANITIZED bytes as its sync baseline, so an unstable rewrite would re-push
  every cycle; a ``[REDACTED:...]`` marker never re-matches any rule.
* **Outbound only.** ``sanitize_inbound_file`` also serves local ``convert``
  writes, where redaction would strip the user's own keys from the converted
  agent and leave it unable to run.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections import Counter, OrderedDict
from typing import NamedTuple

from ms_agent.utils.logger import get_logger
from ._workspace import (scrub_json_secrets, scrub_toml_secrets,
                         scrub_yaml_secrets)

logger = get_logger()

__all__ = ['Finding', 'redact_outbound', 'redact_text']


class Finding(NamedTuple):
    """One redacted secret.

    Carries the *kind* and the key name, never the secret text itself, so the
    upload report and the watch log cannot leak what they are reporting.
    ``line`` is 1-based; Tier A (structural config cleaning) reports line 0
    because a structural scrub has no single location.
    """
    rel: str
    kind: str
    line: int
    name: str


# ---------------------------------------------------------------------------
# Tier B: high-confidence secret VALUES
# ---------------------------------------------------------------------------

# Vendor-prefixed credentials. The prefix is decisive on its own, so only the
# placeholder / variable-reference gate applies to these matches -- a real key
# may legitimately have no digit or an unusual charset. The lookbehind keeps
# ``sk-`` from firing inside ordinary words (``task-tracking``,
# ``<task-notification>``).
_VENDOR_TOKEN_RE = re.compile(
    r'(?<![A-Za-z0-9_\-])(?:'
    r'sk-(?:ant-|proj-|svc-|or-v1-)?[A-Za-z0-9_\-]{12,}'  # OpenAI/Anthropic/
    r'|gh[pousr]_[A-Za-z0-9]{12,}'  # OpenRouter/DashScope
    r'|github_pat_[A-Za-z0-9_]{12,}'  # GitHub
    r'|glpat-[A-Za-z0-9_\-]{12,}'  # GitLab
    r'|xox[baprs]-[A-Za-z0-9\-]{8,}'  # Slack
    r'|AKIA[0-9A-Z]{16}'  # AWS access key id
    r'|AIza[0-9A-Za-z_\-]{20,}'  # Google
    r'|hf_[A-Za-z0-9]{20,}'  # HuggingFace
    r'|npm_[A-Za-z0-9]{20,}'  # npm
    r'|shpat_[A-Za-z0-9]{20,}'  # Shopify
    r')')

_JWT_RE = re.compile(r'(?<![A-Za-z0-9_.\-])'
                     r'eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}'
                     r'\.[A-Za-z0-9_\-]{6,}')

# Case-SENSITIVE on purpose: lowercase "bearer" appears in prose
# ("bearer credentials", "a bearer endpoint") and must not trigger a scan of
# the following word.
_BEARER_RE = re.compile(r'(?<![A-Za-z])Bearer[ \t]+'
                        r'(?P<token>[A-Za-z0-9._~+/=\-]{8,})')

# ``NAME: VALUE`` / ``NAME=VALUE`` with optional quoting on either side, so it
# covers JSON, YAML, shell exports, Python assignments and markdown lists.
_ASSIGN_RE = re.compile(r'(?P<vq1>["\']?)'
                        r'(?P<name>[A-Za-z0-9_][A-Za-z0-9_.\-]{1,62})'
                        r'(?P<vq2>["\']?)'
                        r'(?P<sep>[ \t]*[:=][ \t]*)'
                        r'(?P<vq3>["\']?)'
                        r'(?P<val>[^\s"\'`,;)\]}>]+)')

# ``--flag VALUE`` / ``--flag=VALUE`` / ``-f VALUE`` on a documented command
# line.
_FLAG_RE = re.compile(
    r'(?<![\w\-])-{1,2}(?P<name>[A-Za-z][A-Za-z0-9_.\-]{1,62})'
    r'(?:[= \t]+(?P<vq>["\']?)'
    r'(?P<val>[^\s"\'`,;)\]}>]+))?')

# Absolute URLs in free text. The charset stops at markdown/link punctuation
# so `[doc](https://h/p)` and trailing ``.,;:`` are not swallowed.
_FREE_URL_RE = re.compile(r'(?<![A-Za-z0-9_])'
                          r'(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)'
                          r'(?P<rest>[^\s`"\'<>()\[\]{},;|\\^]+)')

# A base64 payload that decodes to a secret ("decode this at runtime" persona
# instructions). Bounded length keeps data URIs and minified bundles out.
_BASE64_RE = re.compile(r'(?<![A-Za-z0-9+/=_\-])'
                        r'(?P<blob>[A-Za-z0-9+/]{20,512}={0,2})'
                        r'(?![A-Za-z0-9+/=])')
_BASE64_MAX_ATTEMPTS = 400

# Values that are documentation placeholders rather than credentials. Only
# unambiguous markers qualify: a 5+ character word, a repeated-character run
# (``xxxx``), or filler digits. Short words (``foo``, ``test``, ``none``) are
# deliberately absent -- they occur inside a random base62 credential often
# enough (~1 in 1000) that treating them as placeholders would let real keys
# through, and a missed key is worse than a redacted doc example.
_PLACEHOLDER_RE = re.compile(
    r'(?:your|yours|xxx+|yyy+|zzz+|example|dummy|fake|sample|placeholder|'
    r'changeme|change_me|redacted|todo|fixme|localhost|something|whatever|'
    r'unknown|mysecret|secretvalue|000000|123456)', re.IGNORECASE)

# Characters that mark a value as an expression / path / reference rather than
# a credential (``keyvaultref:$SECRET_URI``, ``options?.pageToken``,
# ``${API_KEY}``, ``/etc/keys/a.pem``).
_BAD_VALUE_CHARS = frozenset('$={}?*()@/\\`|&<>:,#%')

# A fully base64-ish value is allowed to carry ``/``, ``+`` and ``=`` padding.
_BASE64ISH_RE = re.compile(r'^[A-Za-z0-9+/]{16,}={0,2}$')

_CAMEL_RE = re.compile(r'[a-z][A-Z]')

# Tier B name vocabulary, in three strengths. Deliberately NARROWER than
# :func:`._workspace.is_secret_key`: bare plural ``tokens`` / ``keys`` and the
# pagination / telemetry spellings in :data:`_NAME_DENY` are ordinary data
# fields (a nanobot ``memory/history.jsonl`` cursor, an LLM ``max_tokens``
# setting) whose values are high-entropy digit-bearing strings -- exactly what
# a credential looks like. ``sk`` is dropped too: it is meaningless as a key
# name and real ``sk-`` values are caught by :data:`_VENDOR_TOKEN_RE`.
#
# STRONG names (``api_key``, ``access_token``, ``client_secret``, ...) are
# unambiguous: a credential is what they hold.
_STRONG_NAME_RE = re.compile(
    r'(?:^|[_\-.])(?:'
    r'api[_-]?keys?|apikeys?|access[_-]?keys?|secret[_-]?keys?|'
    r'client[_-]?secrets?|access[_-]?tokens?|auth[_-]?tokens?|'
    r'id[_-]?tokens?|refresh[_-]?tokens?|token[_-]?file|'
    r'authorization|passphrases?|private[_-]?keys?|'
    r'auth[_-]?key|secret[_-]?key|master[_-]?key|signing[_-]?key)$',
    re.IGNORECASE)

# WEAK names are BARE singulars. They do hold credentials in a memory note
# ("gateway token: ...") or a documented command line (``--token ...``), but
# they are also ordinary data-mapping fields (``{"key": "user_profile_2"}``),
# so they only fire against a stricter value gate. The same word WITH a
# qualifier (``github_token``, ``db_password``, ``smtp_passwd``) is
# unambiguous and counts as strong.
_WEAK_NAME_RE = re.compile(
    r'(?:^|[_\-.])(?:keys?|tokens?|secrets?|passwords?|passwd|'
    r'credentials?|cookies?|bearer)$', re.IGNORECASE)

# Bare stems, singularized: a name whose normalized form IS one of these (and
# nothing more) is weak.
_WEAK_BARE_STEMS = frozenset((
    'key',
    'token',
    'secret',
    'password',
    'passwd',
    'credential',
    'cookie',
    'bearer',
))

# Names that are never a trigger. Plurals are data fields (an LLM
# ``max_tokens`` setting, a nanobot ``memory/history.jsonl`` cursor) whose
# values are high-entropy digit-bearing strings -- exactly what a credential
# looks like -- and the metadata spellings describe a secret without being
# one. ``sk`` is absent from every vocabulary for the same reason: it is
# meaningless as a key name, and real ``sk-`` values are caught by
# :data:`_VENDOR_TOKEN_RE`.
_NAME_DENY = frozenset((
    # pagination / telemetry cursors
    'page_token',
    'next_token',
    'next_page_token',
    'continuation_token',
    'cursor',
    'page_cursor',
    'session_id',
    'request_id',
    'trace_id',
    'span_id',
    'correlation_id',
    # plural data fields
    'tokens',
    'keys',
    'secrets',
    'credentials',
    'cookies',
    # credential metadata
    'key_id',
    'key_name',
    'key_type',
    'key_path',
    'key_alias',
    'token_type',
    'token_name',
    'secret_name',
    'secret_id',
))

# Name strength: 0 = never a trigger, 1 = weak (strict value gate),
# 2 = strong.
_DENY, WEAK, STRONG = 0, 1, 2


def name_strength(name: str) -> int:
    """How much a key / flag *name* is allowed to drive Tier B redaction.

    ``name`` may be dotted or dashed (``model.api_key``, ``--auth-token``).
    An explicit strong spelling wins first (``client_secrets`` and
    ``access_tokens`` are credentials even though their last segment is a
    denied plural), then the deny list, then the generic vocabulary -- where a
    BARE singular is weak and a qualified one (``github_token``,
    ``db_password``) is strong. Testing the last segment is what keeps
    ``options.pageToken`` from being a trigger at all: there is no separator
    before ``token``.
    """
    full = name.strip().strip('\'"').lstrip('-').lower()
    if not full:
        return _DENY
    normalized = full.replace('-', '_').replace('.', '_')
    if _STRONG_NAME_RE.search(full):
        return STRONG
    if normalized in _NAME_DENY or normalized.split('_')[-1] in _NAME_DENY:
        return _DENY
    if _WEAK_NAME_RE.search(full):
        if normalized.rstrip('s') in _WEAK_BARE_STEMS:
            return WEAK
        return STRONG
    return _DENY


def _shannon_entropy(value: str) -> float:
    """Bits per character of *value* (0.0 for the empty string)."""
    if not value:
        return 0.0
    total = len(value)
    return -sum(
        (c / total) * math.log2(c / total) for c in Counter(value).values())


# Lowercase snake_case / kebab-case is how a data-mapping value or a doc
# placeholder spells itself (``user_profile_2``, ``my-super-secret-1``); real
# credentials are hex, base64 or mixed-case. Only applied to WEAK names, so a
# strongly named ``api_key`` keeps its value whatever the casing.
_SNAKE_CASE_RE = re.compile(r'^[a-z0-9]+(?:[_\-][a-z0-9]+)+$')

# Letter-words with at most TRAILING digits: the shape of a variable or field
# name (``SendGridApiKey``, ``MySecretValue123``, ``options.pageToken``), not
# of a credential. A real key interleaves digits with letters
# (``S42bMemTokenLeak01``), which fails this shape and stays eligible.
_IDENTIFIER_SHAPE_RE = re.compile(r'^[A-Za-z][A-Za-z_]*[0-9]*$')


def _looks_real(value: str, quoted: bool, strength: int = STRONG) -> bool:
    """Whether *value* looks like a real credential rather than a placeholder.

    This gate is what keeps the redactor from eating documentation. A value
    must be long enough, carry at least one digit, be high-entropy, and must
    not be a variable reference, a placeholder word, an expression, a path, or
    a camelCase identifier (``SendGridApiKey``, ``options.pageToken``). Fully
    base64-shaped values are exempt from the punctuation ban and the identifier
    test, since real keys legitimately carry ``/``, ``+`` and ``=`` padding.

    A WEAK name (bare ``key`` / ``token`` / ``secret`` / ...) raises every
    threshold and additionally refuses lowercase snake_case values.
    """
    val = value.strip().strip('\'"`')
    if not val:
        return False
    if val.startswith(('[', '<', '{', '$', '%', '#', '-', '+')):
        return False
    if '[REDACTED' in val:
        return False
    base64ish = bool(_BASE64ISH_RE.match(val))
    if not base64ish:
        stripped = val.rstrip('=')
        if any(c in _BAD_VALUE_CHARS for c in stripped):
            return False
        if _CAMEL_RE.search(val) and _IDENTIFIER_SHAPE_RE.match(val):
            return False
        if strength == WEAK and _SNAKE_CASE_RE.match(val):
            return False
    if _PLACEHOLDER_RE.search(val):
        return False
    if strength == WEAK:
        min_len, min_entropy = (14, 3.3) if quoted else (16, 3.3)
    else:
        min_len, min_entropy = (12, 3.0) if quoted else (16, 3.2)
    if len(val) < min_len:
        return False
    if len(set(val)) <= 3:
        return False
    if not any(c.isdigit() for c in val):
        return False
    return _shannon_entropy(val) >= min_entropy


def _marker(kind: str) -> str:
    return f'[REDACTED:{kind}]'


def _line_of(text: str, pos: int) -> int:
    return text.count('\n', 0, pos) + 1


def _redact_vendor(text: str, hits: list) -> str:

    def repl(m):
        value = m.group(0)
        if _PLACEHOLDER_RE.search(value):
            return value
        hits.append(('api_key', _line_of(text, m.start()), ''))
        return _marker('api_key')

    return _VENDOR_TOKEN_RE.sub(repl, text)


def _redact_jwt(text: str, hits: list) -> str:

    def repl(m):
        hits.append(('jwt', _line_of(text, m.start()), ''))
        return _marker('jwt')

    return _JWT_RE.sub(repl, text)


def _redact_bearer(text: str, hits: list) -> str:

    def repl(m):
        token = m.group('token')
        if not _looks_real(token, quoted=False):
            return m.group(0)
        hits.append(('bearer', _line_of(text, m.start('token')), ''))
        return f'Bearer {_marker("bearer")}'

    return _BEARER_RE.sub(repl, text)


def _redact_urls(text: str, hits: list) -> str:
    """Strip credentials from URLs embedded in free text.

    Lenient sibling of :func:`._workspace.scrub_url_secrets`, which is written
    for config scalars and blanks a secret-named query parameter even when its
    value is a variable reference (``?access_token=${token}`` in a converted
    ``AGENTS.md``). Here every credential candidate passes :func:`_looks_real`
    first, so documentation examples such as
    ``postgresql+asyncpg://user:pass@host`` survive.
    """

    def repl(m):
        rest = m.group('rest')
        trail = ''
        while rest and rest[-1] in '.,;:!?\'"':
            trail = rest[-1] + trail
            rest = rest[:-1]
        if not rest:
            return m.group(0)
        cut = len(rest)
        for ch in ('/', '?', '#'):
            idx = rest.find(ch)
            if idx != -1:
                cut = min(cut, idx)
        authority, tail = rest[:cut], rest[cut:]
        changed = False
        at = authority.rfind('@')
        if at != -1:
            userinfo = authority[:at]
            user, colon, password = userinfo.partition(':')
            if colon and password and _looks_real(password, quoted=False):
                authority = (f'{user}:{_marker("password")}@'
                             + authority[at + 1:])
                changed = True
                hits.append(('password', _line_of(text,
                                                  m.start()), 'userinfo'))
            elif not colon and userinfo and _looks_real(
                    userinfo, quoted=False):
                # Colon-less userinfo is how PAT-style tokens travel
                # (``https://ghp_xxx@host``) and cannot be told from a
                # username: fail closed, as ``scrub_url_secrets`` does.
                authority = f'{_marker("token")}@' + authority[at + 1:]
                changed = True
                hits.append(('token', _line_of(text, m.start()), ''))
        if '?' in tail:
            path, query = tail.split('?', 1)
            fragment = ''
            if '#' in query:
                query, frag = query.split('#', 1)
                fragment = '#' + frag
            pairs = []
            for pair in query.split('&'):
                name, eq, value = pair.partition('=')
                strength = name_strength(name) if eq else _DENY
                if strength > _DENY and _looks_real(
                        value, quoted=True, strength=strength):
                    pairs.append(f'{name}={_marker("api_key")}')
                    changed = True
                    hits.append(('api_key', _line_of(text, m.start()), name))
                else:
                    pairs.append(pair)
            tail = path + '?' + '&'.join(pairs) + fragment
        if not changed:
            return m.group(0)
        return f'{m.group("scheme")}{authority}{tail}{trail}'

    return _FREE_URL_RE.sub(repl, text)


def _redact_assignments(text: str, hits: list) -> str:

    def repl(m):
        value = m.group('val')
        quoted = m.group('vq3') in ('"', "'")
        strength = name_strength(m.group('name'))
        if strength == _DENY:
            return m.group(0)
        if not _looks_real(value, quoted=quoted, strength=strength):
            return m.group(0)
        hits.append(('credential', _line_of(text,
                                            m.start('val')), m.group('name')))
        return (f"{m.group('vq1')}{m.group('name')}{m.group('vq2')}"
                f"{m.group('sep')}{m.group('vq3')}{_marker('credential')}"
                f"{m.group('vq3')}")

    return _ASSIGN_RE.sub(repl, text)


def _redact_flags(text: str, hits: list) -> str:

    def repl(m):
        value = m.group('val')
        if value is None:
            return m.group(0)
        quoted = m.group('vq') in ('"', "'")
        strength = name_strength(m.group('name'))
        if strength == _DENY:
            return m.group(0)
        if not _looks_real(value, quoted=quoted, strength=strength):
            return m.group(0)
        hits.append(('flag', _line_of(text, m.start('val')), m.group('name')))
        return (f"--{m.group('name')}={m.group('vq')}{_marker('flag')}"
                f"{m.group('vq')}")

    return _FLAG_RE.sub(repl, text)


def _redact_base64(text: str, hits: list) -> str:
    """Replace a base64 payload whose DECODED content is a secret.

    Persona files sometimes carry "decode this at runtime" instructions whose
    plaintext never appears in the file. Only blobs that decode to valid
    printable text matching a credential rule are touched, so hashes (invalid
    UTF-8), data URIs and ordinary identifiers survive.
    """
    attempts = 0

    def repl(m):
        nonlocal attempts
        blob = m.group('blob')
        start = m.start('blob')
        if attempts >= _BASE64_MAX_ATTEMPTS:
            return m.group(0)
        attempts += 1
        if text[max(0, start - 5):start].endswith('data:'):
            return m.group(0)
        try:
            decoded = base64.b64decode(blob, validate=True)
        except (binascii.Error, ValueError):
            return m.group(0)
        try:
            plain = decoded.decode('utf-8')
        except UnicodeDecodeError:
            return m.group(0)
        if not plain or any(c in plain for c in '\x00\n\r\t'):
            return m.group(0)
        inner: list = []
        scanned = _redact_vendor(plain, inner)
        scanned = _redact_jwt(scanned, inner)
        if not inner:
            assign_hits: list = []
            scanned = _redact_assignments(plain, assign_hits)
            if not assign_hits:
                return m.group(0)
            inner = assign_hits
        hits.append(('base64', _line_of(text, start), inner[0][2]))
        return _marker('base64')

    return _BASE64_RE.sub(repl, text)


# One cheap union of every Tier B textual trigger. A file that matches none of
# these cannot be redacted by any of the gated passes, so skipping them is
# lossless -- and the watch daemon re-runs this over the whole workspace every
# poll. Deliberately over-matches (``monkey`` contains ``key``): it is a gate,
# not a rule. The base64 pass is NOT gated, because an encoded payload carries
# no textual signal at all.
_PREFILTER_RE = re.compile(
    r'(?:sk-|gh[pousr]_|github_pat_|glpat-|xox[baprs]-|AKIA|AIza|hf_|npm_|'
    r'shpat_|eyJ|[Bb]earer|://|'
    r'key|keys|token|tokens|secret|secrets|password|passwd|credential|'
    r'credentials|authorization|cookie|cookies)', re.IGNORECASE)


def redact_text(text: str) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    """Redact high-confidence secret values in free text (Tier B).

    Returns ``(text, hits)`` where each hit is ``(kind, line, name)``. The
    input is returned unchanged (the same ``str`` object) when nothing
    matched. Passes run in a fixed order -- base64, vendor prefixes, JWT,
    ``Bearer``, URLs, ``NAME=VALUE``, ``--flag VALUE`` -- so a secret is
    replaced by the most specific rule that sees it first and the later passes
    only ever encounter a ``[REDACTED:...]`` marker, which no rule matches.
    """
    if not text:
        return text, ()
    hits: list[tuple[str, int, str]] = []
    out = _redact_base64(text, hits)
    if _PREFILTER_RE.search(text):
        out = _redact_vendor(out, hits)
        out = _redact_jwt(out, hits)
        out = _redact_bearer(out, hits)
        out = _redact_urls(out, hits)
        out = _redact_assignments(out, hits)
        out = _redact_flags(out, hits)
    if out == text:
        return text, ()
    return out, tuple(hits)


# ---------------------------------------------------------------------------
# Tier A: config-shaped files at ANY path
# ---------------------------------------------------------------------------

_JSON_CONFIG_NAMES = frozenset(('mcp.json', 'settings.json', 'agent.json'))
_YAML_CONFIG_NAMES = frozenset(
    ('config.yaml', 'config.yml', 'mcp.yaml', 'mcp.yml'))
_TOML_CONFIG_NAMES = frozenset(('config.toml', 'mcp.toml'))
_MCP_JSON_SUFFIXES = ('.mcp.json', )

_MCP_ROOT_KEYS = frozenset(('mcpServers', 'mcp_servers', 'mcpservers'))

_YAML_MCP_SHAPE_RE = re.compile(
    r'^[ \t]*["\']?(?:mcpServers|mcp_servers)'
    r'["\']?[ \t]*:', re.MULTILINE)
_TOML_MCP_SHAPE_RE = re.compile(
    r'(?:^[ \t]*\[{1,2}[^\]\n]*\b(?:mcpServers|mcp_servers)\b'
    r'|^[ \t]*(?:mcpServers|mcp_servers)[ \t]*=)', re.MULTILINE)


def _json_has_mcp(obj) -> bool:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _MCP_ROOT_KEYS and isinstance(val, dict):
                return True
            if _json_has_mcp(val):
                return True
    elif isinstance(obj, list):
        return any(_json_has_mcp(item) for item in obj)
    return False


def _scrub_json_mcp_subtrees(obj) -> bool:
    """Clean only the ``mcpServers`` subtrees of a parsed JSON document.

    A skill's ``mcp.json``-shaped payload embedded in an otherwise unrelated
    JSON document gets the full structural policy (``env`` / ``headers`` bags
    cleared, ``args`` command lines scrubbed, URL credentials stripped) while
    the rest of the document -- which may legitimately hold ``tokens`` or
    ``keys`` data fields -- is left alone. Returns whether anything changed.
    """
    changed = False
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in _MCP_ROOT_KEYS and isinstance(val, dict):
                before = json.dumps(val, sort_keys=True, ensure_ascii=False)
                scrub_json_secrets(val)
                if json.dumps(
                        val, sort_keys=True, ensure_ascii=False) != before:
                    changed = True
            elif _scrub_json_mcp_subtrees(val):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _scrub_json_mcp_subtrees(item):
                changed = True
    return changed


def _redact_config(rel_path: str, text: str, hits: list[tuple[str, int,
                                                              str]]) -> str:
    """Tier A: structural cleaning of a config-shaped file at any path.

    Best effort by design. A known config name that fails to parse falls
    through to Tier B instead of raising: refusing to upload a skill's JSON
    data fixture would be worse than redacting the secret values inside it.
    (The per-framework hook already fails closed for the ROOT config files it
    owns, and it runs before this layer.)
    """
    base = rel_path.rsplit('/', 1)[-1].lower()
    if base.endswith('.json') or base.endswith('.jsonl'):
        is_known = base in _JSON_CONFIG_NAMES or base.endswith(
            _MCP_JSON_SUFFIXES)
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            return text
        if is_known:
            before = json.dumps(data, sort_keys=True, ensure_ascii=False)
            scrub_json_secrets(data)
            if json.dumps(data, sort_keys=True, ensure_ascii=False) == before:
                return text
            hits.append(('config', 0, base))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if _json_has_mcp(data) and _scrub_json_mcp_subtrees(data):
            hits.append(('config', 0, base))
            return json.dumps(data, ensure_ascii=False, indent=2)
        return text
    if base.endswith(('.yaml', '.yml')):
        if base not in _YAML_CONFIG_NAMES \
                and not _YAML_MCP_SHAPE_RE.search(text):
            return text
        cleaned = scrub_yaml_secrets(text)
        if cleaned != text:
            hits.append(('config', 0, base))
        return cleaned
    if base.endswith('.toml'):
        if base not in _TOML_CONFIG_NAMES \
                and not _TOML_MCP_SHAPE_RE.search(text):
            return text
        cleaned = scrub_toml_secrets(text)
        if cleaned != text:
            hits.append(('config', 0, base))
        return cleaned
    return text


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

# Bounded sha256 -> result memo. The watcher re-sanitizes the whole workspace
# every poll and upload re-runs it per invocation, so identical content is
# scanned once. Capped on both entry count and per-file size to keep the
# daemon's footprint flat.
_MEMO_MAX_ENTRIES = 128
_MEMO_MAX_FILE_SIZE = 128 * 1024
_memo: 'OrderedDict[tuple[str, str], tuple[bytes, tuple[Finding, ...]]]' = \
    OrderedDict()


def _redact(rel_path: str, raw: bytes) -> tuple[bytes, tuple[Finding, ...]]:
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        # Binary asset (image, PDF, archive): not a text-secret carrier.
        return raw, ()

    hits: list[tuple[str, int, str]] = []
    out = _redact_config(rel_path, text, hits)
    out, text_hits = redact_text(out)
    hits.extend(text_hits)
    if not hits:
        # Byte identity matters: see the module contract.
        return raw, ()
    return out.encode('utf-8'), tuple(
        Finding(rel_path, kind, line, name) for kind, line, name in hits)


def redact_outbound(rel_path: str,
                    raw: bytes) -> tuple[bytes, tuple[Finding, ...]]:
    """Redact secrets in one collected file, by content rather than by path.

    Runs after the framework's own :meth:`sanitize_outbound_file` hook. Never
    raises: an unparseable, oversized or exotic file is returned unchanged
    (Tier B still gets a best-effort pass at its text), because a crashed
    sanitize would either block the upload or -- inside the watch daemon,
    which swallows exceptions -- silently stop syncing.
    """
    try:
        digest = hashlib.sha256(raw).hexdigest()
        memo_key = (rel_path, digest)
        cached = _memo.get(memo_key)
        if cached is not None:
            _memo.move_to_end(memo_key)
            cleaned, hits = cached
            # The cached bytes came from an earlier call, so hand back THIS
            # caller's object when nothing was redacted: the contract is to
            # return the original, not merely an equal copy.
            return (raw if cleaned == raw else cleaned), hits
        result = _redact(rel_path, raw)
        if len(raw) <= _MEMO_MAX_FILE_SIZE:
            _memo[memo_key] = result
            while len(_memo) > _MEMO_MAX_ENTRIES:
                _memo.popitem(last=False)
        return result
    except Exception:
        logger.debug(
            'Secret redaction failed for %s; uploading as-is.',
            rel_path,
            exc_info=True)
        return raw, ()
