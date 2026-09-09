"""Shared wildcard matching for permission rules.

Rule format: ``server---tool`` or ``server---tool:content_pattern``
Supports ``*`` / ``?`` wildcards via fnmatch, ``|`` to separate alternatives.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlsplit

from ms_agent.utils.pattern_matcher import match_pattern

TOOL_SPLITER = '---'
CONTENT_SEP = ':'
_DOMAIN_PREFIXES = ('domain:', 'domain=', 'url-domain:')
_COMPOUND_RE = re.compile(r'(\|\||&&|;|\n|\|)')


def is_compound_shell(command: str) -> bool:
    """True when a shell string has operators that a prefix glob must not cover."""
    return bool(command and _COMPOUND_RE.search(command))


def _extract_content(tool_name: str, tool_args: dict[str, Any]) -> str | None:
    """Extract the primary content string from tool args for content-pattern matching."""
    val = None
    if tool_name.endswith(f'{TOOL_SPLITER}shell_executor'):
        val = tool_args.get('command')
    elif tool_name.endswith(f'{TOOL_SPLITER}write_file'):
        val = tool_args.get('path')
    elif tool_name.endswith(f'{TOOL_SPLITER}read_file'):
        val = tool_args.get('path')
    elif tool_name.endswith(f'{TOOL_SPLITER}edit_file'):
        val = tool_args.get('path')
    elif tool_name.endswith(f'{TOOL_SPLITER}grep'):
        val = tool_args.get('pattern')
    elif tool_name.endswith(f'{TOOL_SPLITER}glob'):
        val = tool_args.get('pattern')
    else:
        for key in ('path', 'command', 'query', 'url', 'pattern'):
            if key in tool_args:
                val = tool_args[key]
                break
    return str(val) if val is not None else None


def _with_bare_command_variants(content_pattern: str) -> str:
    """Add an argument-less variant for every ``<cmd> *`` alternative.

    ``curl *`` means "the curl command with any arguments" — and running it with
    NONE is a case of that. fnmatch disagrees: it wants the space and at least
    one character after it, so bare ``curl`` slipped past a rule written to gate
    exactly that, and a remembered ``whoami *`` failed to match the very
    ``whoami`` it was generated from.

    Only the space-star idiom of shell commands is extended. Path patterns end
    in ``/*`` (``~/.ssh/*``) or ``=*`` (``dd if=*``) and are left alone — there
    the trailing component is meaningful, not an optional argument list.
    """
    alts = [a.strip() for a in content_pattern.split('|')]
    out = list(alts)
    for alt in alts:
        if alt.endswith(' *'):
            out.append(alt[:-2].rstrip())
    return '|'.join(p for p in out if p)


def normalize_domain(domain: str) -> str:
    """Return a lower-case IDNA ASCII hostname without a trailing dot."""
    domain = domain.strip().rstrip('.').lower()
    if not domain:
        return ''
    try:
        return domain.encode('idna').decode('ascii')
    except UnicodeError:
        return ''


def trusted_url_host(url: str) -> str | None:
    """Extract a public web host suitable for persistent domain trust."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in ('http', 'https'):
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        host = normalize_domain(parsed.hostname or '')
        if not host or host == 'localhost' or host.endswith('.localhost'):
            return None
        private_suffixes = (
            '.local', '.internal', '.lan', '.corp', '.home', '.invalid')
        if any(host.endswith(suffix) for suffix in private_suffixes):
            return None
        blocked_hosts = {
            'metadata',
            'metadata.google.internal',
            'kubernetes',
            'kubernetes.default',
            'kubernetes.default.svc',
        }
        if host in blocked_hosts:
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # Browsers and libc also accept integer/hex/abbreviated IPv4
            # spellings (for example 2130706433 == 127.0.0.1).
            try:
                packed = socket.inet_aton(host)
            except OSError:
                return host
            address = ipaddress.ip_address(packed)
            if not address.is_global:
                return None
            return str(address)
        else:
            if not address.is_global:
                return None
            return host
    except (TypeError, ValueError):
        return None


def _match_domain_rule(rule: str, url: str) -> bool:
    host = trusted_url_host(url)
    if host is None:
        return False
    raw_rule = rule.strip()
    if raw_rule.startswith('*.'):
        suffix = normalize_domain(raw_rule[2:])
        return bool(suffix and host.endswith(f'.{suffix}'))
    wanted = normalize_domain(raw_rule)
    return bool(wanted and host == wanted)


class PermissionMatcher:
    """Wildcard matcher for permission rules, shared by both SafetyGuard and PermissionEnforcer."""

    def match(self, pattern: str, tool_call: str) -> bool:
        """Match a tool call string against a pattern using fnmatch.

        Supports ``|`` separated alternatives: ``read_file|write_file``.
        """
        return match_pattern(pattern, tool_call)

    def match_with_content(
        self,
        pattern: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> bool:
        """Match with optional content pattern after ``:``.

        Examples::

            "file_system---read_file"               → matches tool name only
            "code_executor---shell_executor:pip *"   → matches tool name + command content
            "file_system---*"                        → wildcard on tool name
        """
        if CONTENT_SEP in pattern:
            tool_pattern, content_pattern = pattern.split(CONTENT_SEP, 1)
        else:
            tool_pattern = pattern
            content_pattern = None

        if not self.match(tool_pattern, tool_name):
            return False

        if content_pattern is None:
            return True

        content = _extract_content(tool_name, tool_args)
        if content is None:
            return False

        for prefix in _DOMAIN_PREFIXES:
            if content_pattern.lower().startswith(prefix):
                domain = content_pattern[len(prefix):]
                return _match_domain_rule(domain, content)

        if (
            tool_name.endswith(f'{TOOL_SPLITER}shell_executor')
            and is_compound_shell(content)
            and any(ch in content_pattern for ch in '*?[')
        ):
            return False

        return self.match(_with_bare_command_variants(content_pattern), content)
