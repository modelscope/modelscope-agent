""" /permission slash command: mode switch + rule CRUD. """
from types import SimpleNamespace

import pytest

from ms_agent.command.builtin.permission_cmds import cmd_permission
from ms_agent.command.router import CommandRouter
from ms_agent.command.types import CommandContext, CommandResultType
from ms_agent.permission.config import PermissionConfig
from ms_agent.permission.memory import PermissionMemory


class _Agent:
    def __init__(self, tmp_path):
        self.llm = object()
        memory = PermissionMemory(project_path=tmp_path)
        self.tool_manager = SimpleNamespace(
            _permission_mode='interactive',
            _permission_enforcer=SimpleNamespace(
                _memory=memory,
                _config=PermissionConfig(mode='interactive'),
                _provider=None,
            ),
        )
        self._modes = []

    def set_permission_mode(self, mode):
        self._modes.append(mode)
        self.tool_manager._permission_mode = mode
        return mode

    def set_permission_decision_provider(self, provider):
        self.tool_manager._permission_enforcer._provider = provider


def _ctx(agent, text):
    router = CommandRouter(owner=agent)
    cmd, args = CommandRouter.parse_input(text)
    return CommandContext(
        raw_input=text,
        command_name=cmd,
        args=args,
        extra={'router': router},
    )


@pytest.mark.asyncio
async def test_permission_list_edit_delete(tmp_path):
    agent = _Agent(tmp_path)
    memory = agent.tool_manager._permission_enforcer._memory
    entry = memory.add('code_executor---shell_executor:echo *')

    listed = await cmd_permission(_ctx(agent, '/permission list'))
    assert entry.id[:8] in listed.content
    assert 'echo *' in listed.content
    assert '/permission edit' in listed.content

    edited = await cmd_permission(
        _ctx(agent, f'/permission edit {entry.id[:8]} '
             'code_executor---shell_executor:echo permission-e2e*'))
    assert 'echo permission-e2e*' in edited.content
    assert memory.list()[0].pattern.endswith('echo permission-e2e*')

    deleted = await cmd_permission(
        _ctx(agent, f'/permission delete {entry.id[:8]}'))
    assert 'Deleted' in deleted.content
    assert memory.list() == []


@pytest.mark.asyncio
async def test_permission_edit_prompts_when_pattern_omitted(
        tmp_path, monkeypatch):
    agent = _Agent(tmp_path)
    memory = agent.tool_manager._permission_enforcer._memory
    entry = memory.add('code_executor---shell_executor:echo *')

    async def fake_prompt(current):
        assert current == entry.pattern
        return 'code_executor---shell_executor:echo hi*'

    monkeypatch.setattr(
        'ms_agent.command.builtin.permission_cmds._prompt_pattern',
        fake_prompt)
    monkeypatch.setattr(
        'ms_agent.command.builtin.permission_cmds._is_tty', lambda: True)

    edited = await cmd_permission(
        _ctx(agent, f'/permission edit {entry.id[:8]}'))
    assert 'echo hi*' in edited.content
    assert memory.list()[0].pattern.endswith('echo hi*')


@pytest.mark.asyncio
async def test_permission_edit_picks_rule_then_prompts(
        tmp_path, monkeypatch):
    agent = _Agent(tmp_path)
    memory = agent.tool_manager._permission_enforcer._memory
    first = memory.add('code_executor---shell_executor:echo *')
    memory.add('code_executor---shell_executor:ping *')

    async def fake_pick(entries):
        assert len(entries) == 2
        return entries[0]

    async def fake_prompt(current):
        assert current == first.pattern
        return 'code_executor---shell_executor:echo picked*'

    monkeypatch.setattr(
        'ms_agent.command.builtin.permission_cmds._pick_rule', fake_pick)
    monkeypatch.setattr(
        'ms_agent.command.builtin.permission_cmds._prompt_pattern',
        fake_prompt)
    monkeypatch.setattr(
        'ms_agent.command.builtin.permission_cmds._is_tty', lambda: True)

    edited = await cmd_permission(_ctx(agent, '/permission edit'))
    assert 'echo picked*' in edited.content
    updated = next(e for e in memory.list() if e.id == first.id)
    assert updated.pattern.endswith('echo picked*')


@pytest.mark.asyncio
async def test_permission_edit_id_only_non_tty_shows_current(tmp_path):
    agent = _Agent(tmp_path)
    memory = agent.tool_manager._permission_enforcer._memory
    entry = memory.add('code_executor---shell_executor:echo *')

    result = await cmd_permission(
        _ctx(agent, f'/permission edit {entry.id[:8]}'))
    assert 'usage:' in result.content
    assert entry.pattern in result.content
    assert memory.list()[0].pattern == entry.pattern


@pytest.mark.asyncio
async def test_permission_edit_rejects_bare_wildcard(tmp_path):
    agent = _Agent(tmp_path)
    memory = agent.tool_manager._permission_enforcer._memory
    entry = memory.add('code_executor---shell_executor:echo *')

    result = await cmd_permission(
        _ctx(agent, f'/permission edit {entry.id[:8]} '
             'code_executor---shell_executor:*'))
    assert 'Unsafe persist edit' in result.content
    assert memory.list()[0].pattern.endswith('echo *')


@pytest.mark.asyncio
async def test_permission_mode_switch_delegate(tmp_path):
    agent = _Agent(tmp_path)
    result = await cmd_permission(_ctx(agent, '/permission delegate'))
    assert result.type == CommandResultType.MUTATE_STATE
    assert 'delegate' in result.content
    assert agent._modes == ['delegate']
