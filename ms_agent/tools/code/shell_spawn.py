"""Isolate shell children from the TUI TTY and reject interactive logins.

Spawned commands must not inherit the agent's controlling terminal (password
prompts steal ↑/↓/Enter). Auth is key / ssh-agent only; a login shell without
a remote command is rejected up front.
"""

from __future__ import annotations

import os
import shlex
from typing import Any

import asyncio.subprocess as ai_subprocess

# ssh options that consume the next argv (man ssh).
_SSH_TAKES_ARG = set('BbcDEeFIiJLlmOopQRSWw')
# These mean "no login shell" even when a destination is given and no
# remote command follows (-N/-f/-O/-W: tunnel/control; -G: dump config;
# -V: print version and exit).
_SSH_NON_LOGIN_OPTS = set('NfOGWV')

_INTERACTIVE_LOGIN_ERROR = (
    'Interactive SSH/SFTP login is not supported (no TTY and no password '
    "prompt). Use a remote command, e.g. ssh user@host 'uname -a'. "
    'Authenticate with keys / ssh-agent; password prompts fail immediately.'
)


def isolated_subprocess_kwargs(**extra: Any) -> dict[str, Any]:
    """Kwargs so the child cannot steal the TUI's stdin/TTY."""
    kwargs: dict[str, Any] = {
        'stdin': ai_subprocess.DEVNULL,
        'stdout': ai_subprocess.PIPE,
        'stderr': ai_subprocess.PIPE,
        'start_new_session': True,
    }
    kwargs.update(extra)
    return kwargs


def interactive_login_error(command: str) -> str | None:
    """If ``command`` is a TTY login (bare ssh/sftp), return an error string."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    i = 0
    while i < len(tokens) and '=' in tokens[i] and not tokens[i].startswith('-'):
        i += 1
    if i >= len(tokens):
        return None
    name = os.path.basename(tokens[i])
    rest = tokens[i + 1:]
    if name in ('ssh', 'ssh.exe'):
        if _ssh_is_login(rest):
            return _INTERACTIVE_LOGIN_ERROR
        return None
    if name in ('sftp', 'sftp.exe'):
        if '-b' not in rest and '/b' not in rest:
            return _INTERACTIVE_LOGIN_ERROR
    return None


def _ssh_is_login(args: list[str]) -> bool:
    positional, opt_chars = _skip_ssh_options(args)
    if opt_chars & _SSH_NON_LOGIN_OPTS:
        return False
    # No destination (ssh, ssh -h, ssh -V, ssh --help) prints usage/version.
    # Destination + command is batch ssh. Destination only is a login shell.
    return len(positional) == 1


def _skip_ssh_options(args: list[str]) -> tuple[list[str], set[str]]:
    seen: set[str] = set()
    i = 0
    n = len(args)
    while i < n:
        a = args[i]
        if a == '--':
            return args[i + 1:], seen
        if not a.startswith('-') or a == '-':
            return args[i:], seen
        if a.startswith('--'):
            i += 1
            continue
        chars = a[1:]
        k = 0
        took_arg = False
        while k < len(chars):
            ch = chars[k]
            seen.add(ch)
            if ch in _SSH_TAKES_ARG:
                attached = chars[k + 1:]
                if not attached:
                    i += 1  # next argv is the option value
                took_arg = True
                break
            k += 1
        i += 1
        if took_arg:
            continue
    return [], seen
