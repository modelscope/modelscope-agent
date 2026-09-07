"""Scene-specific one-layer permission options (Claude Code style).

Each AskOption is a complete decision: choosing a row allows once, persists a
bound pattern, or denies. There is no nested "Always allow → pick a template"
step. Editing a persist prefix happens on the persist row itself.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal, Sequence

from .handler import PermissionAction, PermissionResponse
from .matcher import CONTENT_SEP, TOOL_SPLITER, is_compound_shell, trusted_url_host
from .suggestions import generate_suggestions

_CHOICE_RE = re.compile(r'^(\d+)$')
_CHOICE_EDIT_RE = re.compile(r'^(\d+)\s*(?:=\s*|\s+)(.+)$')

Scope = Literal['project', 'global']


@dataclass(frozen=True)
class AskOption:
    """One row in the permission prompt."""

    key: Literal['yes', 'persist', 'no']
    label: str
    action: PermissionAction
    pattern: str | None = None
    scope: Scope = 'project'
    editable: bool = False
    edit_value: str = ''
    pattern_prefix: str = ''

    def with_edit(self, edit_value: str) -> 'AskOption':
        text = edit_value.strip()
        if not text or not self.editable:
            return self
        if '|' in text or text in ('*', '?', '**'):
            raise ValueError(f'Unsafe persist edit: {text!r}')
        pattern = f'{self.pattern_prefix}{text}' if self.pattern_prefix else text
        label = _label_with_scope(self.label, self.edit_value, text)
        return AskOption(
            key=self.key,
            label=label,
            action=self.action,
            pattern=pattern,
            scope=self.scope,
            editable=True,
            edit_value=text,
            pattern_prefix=self.pattern_prefix,
        )

    def to_response(self) -> PermissionResponse:
        return PermissionResponse(
            action=self.action,
            pattern=self.pattern,
            scope=self.scope,
        )


def build_ask_options(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    workspace_root: str | os.PathLike[str] | None = None,
    suggestions: Sequence[str] | None = None,
) -> list[AskOption]:
    """Return Yes / scene persist / No. Persist is omitted when unsafe."""

    persist = _persist_option(
        tool_name,
        tool_args,
        workspace_root=workspace_root,
        suggestions=suggestions,
    )
    options = [
        AskOption(
            key='yes',
            label='Yes',
            action=PermissionAction.ALLOW_ONCE,
        ),
    ]
    if persist is not None:
        options.append(persist)
    options.append(
        AskOption(
            key='no',
            label='No',
            action=PermissionAction.DENY,
        ),
    )
    return options


def format_ask_header(
    *,
    tool_name: str,
    args_preview: str = '',
    context: str = '',
) -> str:
    """Tool/args context above the option list (TUI select header)."""

    lines = [f'⚠ allow this tool call?  {tool_name}']
    if args_preview:
        lines.append(args_preview)
    if context:
        lines.append(str(context))
    return '\n'.join(lines)


def format_ask_menu(
    options: Sequence[AskOption],
    *,
    tool_name: str,
    args_preview: str = '',
    context: str = '',
) -> str:
    """Human-readable one-layer menu for CLI / non-TTY (also e2e)."""

    lines = [format_ask_header(
        tool_name=tool_name,
        args_preview=args_preview,
        context=context,
    ), '']
    for i, option in enumerate(options, start=1):
        suffix = '  (editable)' if option.editable else ''
        lines.append(f'  {i}. {option.label}{suffix}')
    return '\n'.join(lines)


def parse_ask_choice(
    raw: str | None,
    options: Sequence[AskOption],
) -> PermissionResponse:
    """Parse one stdin line into a decision. EOF / empty-cancel → deny.

    Accepts ``1``, ``2``, ``3``, ``y``/``n``, and an in-line edit
    ``2=echo permission-e2e*`` / ``2 echo permission-e2e*``. Never prompts again.
    """

    if raw is None:
        return PermissionResponse(action=PermissionAction.DENY)
    text = raw.strip()
    if not text:
        return PermissionResponse(action=PermissionAction.DENY)
    lowered = text.lower()
    if lowered in ('n', 'no', 'esc'):
        return options[-1].to_response()
    if lowered in ('y', 'yes'):
        return options[0].to_response()

    edit_match = _CHOICE_EDIT_RE.match(text)
    bare_match = _CHOICE_RE.match(text)
    match = edit_match or bare_match
    if match is None:
        return PermissionResponse(action=PermissionAction.DENY)

    index = int(match.group(1)) - 1
    extra = match.group(2).strip() if edit_match else ''
    if index < 0 or index >= len(options):
        return PermissionResponse(action=PermissionAction.DENY)

    option = options[index]
    if extra and option.editable:
        try:
            option = option.with_edit(extra)
        except ValueError:
            return PermissionResponse(action=PermissionAction.DENY)
    return option.to_response()


def display_rule(pattern: str) -> str:
    """Shorten an internal pattern for menu labels."""

    if f'{CONTENT_SEP}domain:' in pattern:
        return pattern.rsplit('domain:', 1)[-1]
    if CONTENT_SEP in pattern:
        return pattern.split(CONTENT_SEP, 1)[1]
    if TOOL_SPLITER in pattern:
        return pattern.rsplit(TOOL_SPLITER, 1)[-1]
    return pattern


def _persist_option(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    workspace_root: str | os.PathLike[str] | None,
    suggestions: Sequence[str] | None,
) -> AskOption | None:
    url = tool_args.get('url')
    if isinstance(url, str) and url:
        return _url_persist(tool_name, url)

    if tool_name.endswith(f'{TOOL_SPLITER}shell_executor'):
        return _shell_persist(tool_name, tool_args, suggestions)

    if _is_file_write_tool(tool_name):
        return _file_persist(tool_name, tool_args, workspace_root)

    return AskOption(
        key='persist',
        label=f"Yes, and don't ask again for {display_rule(tool_name)}",
        action=PermissionAction.ALLOW_ALWAYS,
        pattern=tool_name,
    )


def _url_persist(tool_name: str, url: str) -> AskOption | None:
    host = trusted_url_host(url)
    if host is None:
        return None
    pattern = f'{tool_name}{CONTENT_SEP}domain:{host}'
    return AskOption(
        key='persist',
        label=f"Yes, and don't ask again for {host}",
        action=PermissionAction.ALLOW_ALWAYS,
        pattern=pattern,
        edit_value=host,
        pattern_prefix=f'{tool_name}{CONTENT_SEP}domain:',
    )


def _shell_persist(
    tool_name: str,
    tool_args: dict[str, Any],
    suggestions: Sequence[str] | None,
) -> AskOption | None:
    command = str(tool_args.get('command', '')).strip()
    if not command:
        return None
    if is_compound_shell(command):
        return None

    patterns = list(suggestions or generate_suggestions(tool_name, tool_args))
    prefix = next(
        (item for item in patterns
         if item.startswith(f'{tool_name}{CONTENT_SEP}') and item.endswith(' *')),
        None,
    )
    if prefix is None:
        return None
    shown = display_rule(prefix)
    return AskOption(
        key='persist',
        label=f"Yes, and don't ask again for {shown}",
        action=PermissionAction.ALLOW_ALWAYS,
        pattern=prefix,
        editable=True,
        edit_value=shown,
        pattern_prefix=f'{tool_name}{CONTENT_SEP}',
    )


def _file_persist(
    tool_name: str,
    tool_args: dict[str, Any],
    workspace_root: str | os.PathLike[str] | None,
) -> AskOption | None:
    path = tool_args.get('path')
    if not isinstance(path, str) or not path:
        return None
    if _path_in_workspace(path, workspace_root):
        return AskOption(
            key='persist',
            label='Yes, allow all edits in this project this session',
            action=PermissionAction.ALLOW_SESSION,
            pattern='file_system---write_file|file_system---edit_file',
        )
    parent = str(PurePath(path).parent)
    if parent in ('', '.'):
        parent = path
    suffix = '' if parent.endswith('/') else '/'
    shown = f'{parent}{suffix}'
    return AskOption(
        key='persist',
        label=f'Yes, allow all edits in {shown} this session',
        action=PermissionAction.ALLOW_SESSION,
        pattern=f'{tool_name}{CONTENT_SEP}{parent}{suffix}*',
        edit_value=shown,
        pattern_prefix=f'{tool_name}{CONTENT_SEP}',
    )


def _is_file_write_tool(tool_name: str) -> bool:
    return tool_name.startswith(f'file_system{TOOL_SPLITER}') and (
        tool_name.endswith('write_file') or tool_name.endswith('edit_file'))


def _path_in_workspace(
    path: str,
    workspace_root: str | os.PathLike[str] | None,
) -> bool:
    if not workspace_root:
        return False
    try:
        resolved = Path(path).expanduser().resolve()
        root = Path(workspace_root).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def _label_with_scope(label: str, old: str, new: str) -> str:
    if old and old in label:
        return label.replace(old, new, 1)
    marker = "don't ask again for "
    if marker in label:
        return label.split(marker, 1)[0] + marker + new
    return label
