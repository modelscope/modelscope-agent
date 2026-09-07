"""Auto-generate permission pattern suggestions for allow_always actions."""

from __future__ import annotations

import shlex
from pathlib import PurePath
from typing import Any

from .matcher import CONTENT_SEP, TOOL_SPLITER, trusted_url_host
from .provider import sanitize_sensitive_text
from .wrapper_strip import strip_safe_wrappers


def generate_suggestions(tool_name: str, tool_args: dict[str,
                                                         Any]) -> list[str]:
    """Generate suggested wildcard patterns based on tool name and arguments.

    Returns a list of patterns from most specific to most general.
    """
    suggestions: list[str] = []

    # Extract server name (everything before first TOOL_SPLITER)
    parts = tool_name.split(TOOL_SPLITER, 1)
    server = parts[0] if len(parts) > 1 else ''

    url = tool_args.get('url')
    if isinstance(url, str) and url:
        host = trusted_url_host(url)
        if host is None:
            return []
        exact = _literal_suggestion(tool_name, url)
        if exact:
            suggestions.append(exact)
        suggestions.append(f'{tool_name}{CONTENT_SEP}domain:{host}')
        suggestions.append(tool_name)
        if server:
            suggestions.append(f'{server}{TOOL_SPLITER}*')
    elif tool_name.endswith(f'{TOOL_SPLITER}shell_executor'):
        command = str(tool_args.get('command', '')).strip()
        if command:
            exact = _literal_suggestion(tool_name, command)
            if exact:
                suggestions.append(exact)
            first_cmd = _extract_first_command(command)
            if first_cmd:
                suggestions.append(f'{tool_name}{CONTENT_SEP}{first_cmd} *')
        suggestions.append(tool_name)
    elif server == 'file_system':
        path = tool_args.get('path')
        if isinstance(path, str) and path:
            exact = _literal_suggestion(tool_name, path)
            if exact:
                suggestions.append(exact)
            parent = str(PurePath(path).parent)
            if parent not in ('', '.'):
                suffix = '' if parent.endswith('/') else '/'
                suggestions.append(
                    f'{tool_name}{CONTENT_SEP}'
                    f'{_escape_glob_literal(parent)}{suffix}*')
        suggestions.append(tool_name)
    elif server == 'web_search':
        suggestions.append(f'{server}{TOOL_SPLITER}*')
    else:
        suggestions.append(tool_name)
        if server:
            suggestions.append(f'{server}{TOOL_SPLITER}*')

    return list(dict.fromkeys(suggestions))


def _literal_suggestion(tool_name: str, value: str) -> str | None:
    """Return an exact pattern, or None when it would be secret or ambiguous."""

    if '|' in value:
        return None
    if sanitize_sensitive_text(value) != value:
        return None
    return f'{tool_name}{CONTENT_SEP}{_escape_glob_literal(value)}'


def _escape_glob_literal(value: str) -> str:
    """Escape fnmatch metacharacters while retaining ordinary characters."""

    return ''.join({
        '*': '[*]',
        '?': '[?]',
        '[': '[[]',
        '|': '[|]',
    }.get(char, char) for char in value)


def _extract_first_command(command: str) -> str:
    """Extract the base command name, stripping safe wrappers (timeout, nice, …)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    stripped = strip_safe_wrappers(tokens)
    return stripped[0] if stripped else ''


