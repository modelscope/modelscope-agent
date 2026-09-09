"""Slash-command handlers for permission mode and saved-rule CRUD."""

from __future__ import annotations

import sys

from ms_agent.command.router import CommandRouter
from ms_agent.command.types import (CommandContext, CommandDef, CommandResult,
                                    CommandResultType)

_MODE_ALIASES = {
    'restricted': 'interactive',
    'delegated': 'delegate',
}
_PUBLIC_MODES = (
    'interactive',
    'delegate',
    'full_access',
    'auto',
    'strict',
)
_USAGE = (
    'usage:\n'
    '  /permission                         show current mode and rules\n'
    '  /permission <interactive|delegate|full_access|auto|strict>\n'
    '  /permission list\n'
    '  /permission edit                    pick a rule, then edit its pattern\n'
    '  /permission edit <id>               edit that rule (prompts for pattern)\n'
    '  /permission edit <id> <pattern>\n'
    '  /permission delete <id>'
)

CMD_PERMISSION = CommandDef(
    name='permission',
    description='Show or switch permission mode; list/edit/delete saved rules',
    category='config',
    aliases=('mode', ),
)


def _agent_from(ctx: CommandContext):
    router = ctx.extra.get('router') if ctx.extra else None
    return getattr(router, 'owner', None)


def _memory(agent):
    tm = getattr(agent, 'tool_manager', None)
    enf = getattr(tm, '_permission_enforcer', None)
    return getattr(enf, '_memory', None)


def _current_mode(agent) -> str:
    tm = getattr(agent, 'tool_manager', None)
    if tm is not None and getattr(tm, '_permission_mode', None):
        return str(tm._permission_mode)
    perm = getattr(getattr(agent, 'config', None), 'permission', None)
    return str(getattr(perm, 'mode', None) or 'auto')


def _format_rules(memory) -> str:
    entries = memory.list()
    if not entries:
        return 'No saved always-allow rules.'
    lines = ['Saved always-allow rules:']
    for entry in entries:
        lines.append(
            f'  {entry.id[:8]}  [{entry.scope}/{entry.kind}]  {entry.pattern}')
    lines.append('  (edit: /permission edit   ·   delete: /permission delete <id>)')
    return '\n'.join(lines)


async def cmd_permission(ctx: CommandContext) -> CommandResult:
    agent = _agent_from(ctx)
    if agent is None:
        return CommandResult(
            type=CommandResultType.MESSAGE,
            content='No active agent.',
        )

    arg = (ctx.args or '').strip()
    if not arg:
        mode = _current_mode(agent)
        memory = _memory(agent)
        body = f'permission mode: {mode}\n{_USAGE}'
        if memory is not None:
            body += '\n\n' + _format_rules(memory)
        return CommandResult(type=CommandResultType.MESSAGE, content=body)

    parts = arg.split(None, 2)
    verb = parts[0].lower()

    if verb in ('list', 'ls'):
        memory = _memory(agent)
        if memory is None:
            return CommandResult(
                type=CommandResultType.MESSAGE,
                content='Permission memory is not initialized yet.',
            )
        return CommandResult(
            type=CommandResultType.MESSAGE, content=_format_rules(memory))

    if verb in ('delete', 'rm', 'revoke'):
        if len(parts) < 2:
            return CommandResult(
                type=CommandResultType.MESSAGE,
                content='usage: /permission delete <id>',
            )
        return CommandResult(
            type=CommandResultType.MESSAGE,
            content=_delete_rule(agent, parts[1]),
        )

    if verb == 'edit':
        return await _cmd_edit(agent, parts)

    mode_token = _MODE_ALIASES.get(verb, verb)
    if mode_token in _PUBLIC_MODES:
        try:
            mode = agent.set_permission_mode(mode_token)
        except ValueError as exc:
            return CommandResult(
                type=CommandResultType.MESSAGE, content=str(exc))
        if mode == 'delegate':
            _ensure_delegate_provider(agent)
        return CommandResult(
            type=CommandResultType.MUTATE_STATE,
            content=f'permission mode → {mode}',
        )

    return CommandResult(
        type=CommandResultType.MESSAGE,
        content=f'Unknown permission command: {verb}\n{_USAGE}',
    )


def _is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


async def _cmd_edit(agent, parts) -> CommandResult:
    memory = _memory(agent)
    if memory is None:
        return CommandResult(
            type=CommandResultType.MESSAGE,
            content='Permission memory is not initialized yet.',
        )
    entries = memory.list()
    if not entries:
        return CommandResult(
            type=CommandResultType.MESSAGE,
            content='No saved always-allow rules.',
        )

    token = parts[1] if len(parts) > 1 else ''
    pattern = parts[2] if len(parts) > 2 else ''

    if not token:
        entry = await _pick_rule(entries)
        if entry is None:
            return CommandResult(
                type=CommandResultType.MESSAGE,
                content=(
                    'Cancelled.' if _is_tty() else
                    'usage: /permission edit <id> <pattern>'),
            )
    else:
        try:
            entry = _match_entry(memory, token)
        except ValueError as exc:
            return CommandResult(
                type=CommandResultType.MESSAGE, content=str(exc))
        except KeyError:
            return CommandResult(
                type=CommandResultType.MESSAGE,
                content=f'No saved rule matching {token!r}.',
            )

    if not pattern:
        pattern = await _prompt_pattern(entry.pattern)
        if not pattern:
            if _is_tty():
                return CommandResult(
                    type=CommandResultType.MESSAGE, content='Cancelled.')
            return CommandResult(
                type=CommandResultType.MESSAGE,
                content=(
                    f'usage: /permission edit {entry.id[:8]} <pattern>\n'
                    f'current: {entry.pattern}'),
            )
    return CommandResult(
        type=CommandResultType.MESSAGE,
        content=_edit_rule(agent, entry.id, pattern),
    )


async def _pick_rule(entries):
    """Arrow-key picker for `/permission edit` with no id."""
    from ms_agent.tui.select import SelectItem, select_async
    from ms_agent.tui.tty import restore_cooked_tty

    items = [
        SelectItem(label=f'{e.id[:8]}  [{e.scope}/{e.kind}]  {e.pattern}')
        for e in entries
    ]
    try:
        result = await select_async(items, header='Select a rule to edit')
    finally:
        restore_cooked_tty()
    if result is None:
        return None
    return entries[result.index]


async def _prompt_pattern(current: str) -> str | None:
    """Pre-filled line edit for the selected rule's pattern."""
    if not _is_tty():
        return None
    from prompt_toolkit import PromptSession

    from ms_agent.tui.tty import restore_cooked_tty

    try:
        session = PromptSession()
        text = await session.prompt_async('new pattern: ', default=current)
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        restore_cooked_tty()
    text = (text or '').strip()
    return text or None


def _match_entry(memory, token: str):
    token = token.strip()
    entries = memory.list()
    exact = [e for e in entries if e.id == token]
    if exact:
        return exact[0]
    prefix = [e for e in entries if e.id.startswith(token)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        raise ValueError(f'Ambiguous rule id {token!r}; pass more characters')
    raise KeyError(token)


def _delete_rule(agent, token: str) -> str:
    memory = _memory(agent)
    if memory is None:
        return 'Permission memory is not initialized yet.'
    try:
        entry = _match_entry(memory, token)
    except ValueError as exc:
        return str(exc)
    except KeyError:
        return f'No saved rule matching {token!r}.'
    memory.delete(entry.id)
    return f'Deleted {entry.id[:8]}  {entry.pattern}'


def _edit_rule(agent, token: str, pattern: str) -> str:
    memory = _memory(agent)
    if memory is None:
        return 'Permission memory is not initialized yet.'
    unsafe = _unsafe_edit_pattern(pattern)
    if unsafe:
        return unsafe
    try:
        entry = _match_entry(memory, token)
        updated = memory.update(entry.id, pattern=pattern)
    except ValueError as exc:
        return str(exc)
    except KeyError:
        return f'No saved rule matching {token!r}.'
    return f'Updated {updated.id[:8]}  {updated.pattern}'


def _unsafe_edit_pattern(pattern: str) -> str | None:
    text = (pattern or '').strip()
    if not text:
        return 'Pattern must not be empty.'
    marker = '---shell_executor:'
    if marker not in text:
        return None
    content = text.split(marker, 1)[1]
    if '|' in content or content in ('*', '?', '**'):
        return f'Unsafe persist edit: {content!r}'
    return None


def _ensure_delegate_provider(agent) -> None:
    from dataclasses import replace

    from ms_agent.permission import LlmDecisionProvider

    tm = getattr(agent, 'tool_manager', None)
    enf = getattr(tm, '_permission_enforcer', None) if tm else None
    if enf is None:
        return
    cfg = getattr(enf, '_config', None)
    if cfg is not None and cfg.decision_provider is None:
        enf._config = replace(cfg, decision_provider='llm')
    if getattr(enf, '_provider', None) is None and getattr(agent, 'llm', None):
        provider = LlmDecisionProvider(agent.llm)
        agent.set_permission_decision_provider(provider)


def register_permission_commands(router: CommandRouter) -> None:
    router.register(CMD_PERMISSION, cmd_permission)
