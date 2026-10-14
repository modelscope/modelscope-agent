# Copyright (c) ModelScope Contributors. All rights reserved.
"""Next-message config sync stays on the same agent."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from ms_agent.tui.live_config import apply_live_config


@pytest.mark.asyncio
async def test_model_change_rebuilds_the_client_in_place():
    config = OmegaConf.create({
        'llm': {
            'service': 'openai',
            'model': 'gpt-4o',
        },
        'generation_config': {
            'stream': True,
        },
    })
    live = SimpleNamespace(model='old', config=config, _setup_stub=False)
    agent = SimpleNamespace(
        config=config,
        llm=live,
        runtime=SimpleNamespace(llm=live),
        memory_tools=[],
        _skill_runtime=None,
        mcp_runtime=None,
        _llm_fp='stale',
        _mcp_fp=None,
        _memory_guidance='',
    )

    class Rebuilt:
        def __init__(self):
            self.model = 'gpt-4o'
            self.config = config

    with patch('ms_agent.llm.LLM.from_config', return_value=Rebuilt()):
        await apply_live_config(agent, None)
    assert agent.llm.model == 'gpt-4o'
    assert agent.runtime.llm.model == 'gpt-4o'
    assert agent._llm_fp != 'stale'


@pytest.mark.asyncio
async def test_unchanged_token_does_not_rebuild():
    from ms_agent.llm.runtime_token import llm_runtime_token
    config = OmegaConf.create({
        'llm': {
            'service': 'openai',
            'model': 'gpt-4o',
        },
    })
    live = SimpleNamespace(model='gpt-4o', config=config, _setup_stub=False)
    agent = SimpleNamespace(
        config=config,
        llm=live,
        runtime=SimpleNamespace(llm=live),
        memory_tools=[],
        _skill_runtime=None,
        mcp_runtime=None,
        _llm_fp=llm_runtime_token(config),
        _mcp_fp=None,
    )
    with patch('ms_agent.llm.LLM.from_config') as rebuild:
        await apply_live_config(agent, None)
    rebuild.assert_not_called()


def _quiet_agent(tmp_path, **extra):
    config = OmegaConf.create({'output_dir': str(tmp_path)})
    agent = SimpleNamespace(
        config=config,
        llm=None,
        runtime=SimpleNamespace(),
        memory_tools=[],
        _skill_runtime=None,
        mcp_runtime=None,
        tool_manager=None,
        _llm_fp=None,
        _mcp_fp=None,
        _search_fp=None,
        _live_notice_keys=set(),
        notices=[],
    )
    agent._emit_user_notice = agent.notices.append
    for key, value in extra.items():
        setattr(agent, key, value)
    return agent


@pytest.mark.asyncio
async def test_bad_json_is_said_once(tmp_path, monkeypatch):
    monkeypatch.setenv('MS_AGENT_HOME', str(tmp_path))
    (tmp_path / 'mcp.json').write_text('{', encoding='utf-8')
    agent = _quiet_agent(tmp_path)
    await apply_live_config(agent, None)
    await apply_live_config(agent, None)
    assert agent.notices == ['mcp.json 读不进去。请检查格式。']


@pytest.mark.asyncio
async def test_pending_mcp_reconnects_when_the_file_is_unchanged(
        tmp_path, monkeypatch):
    monkeypatch.setenv('MS_AGENT_HOME', str(tmp_path))
    from ms_agent.tui.live_config import _token

    class Runtime:
        def __init__(self):
            self.calls = 0

        async def apply_config(self, config):
            del config
            self.calls += 1

        def get_server(self, name):
            del name
            return SimpleNamespace(status='connected')

    runtime = Runtime()
    agent = _quiet_agent(
        tmp_path,
        mcp_runtime=runtime,
        _mcp_fp=_token({}),
        _mcp_reconnect_names={'fetch'},
    )
    await apply_live_config(agent, None)
    await apply_live_config(agent, None)
    assert runtime.calls == 1
    assert agent.notices == []
    assert agent._mcp_reconnect_names == set()


@pytest.mark.asyncio
async def test_failed_mcp_reconnect_is_said_once(tmp_path, monkeypatch):
    monkeypatch.setenv('MS_AGENT_HOME', str(tmp_path))
    from ms_agent.tui.live_config import _token

    class Runtime:
        def __init__(self):
            self.calls = 0

        async def apply_config(self, config):
            del config
            self.calls += 1

        def get_server(self, name):
            del name
            return SimpleNamespace(status='error')

    runtime = Runtime()
    agent = _quiet_agent(
        tmp_path,
        mcp_runtime=runtime,
        _mcp_fp=_token({}),
        _mcp_reconnect_names={'fetch'},
    )
    await apply_live_config(agent, None)
    await apply_live_config(agent, None)
    assert runtime.calls == 1
    assert agent.notices == ['MCP 服务器 fetch 还是连不上。']


@pytest.mark.asyncio
async def test_search_engine_reloads_on_the_next_message(tmp_path, monkeypatch):
    monkeypatch.setenv('MS_AGENT_HOME', str(tmp_path))
    from ms_agent.config.search_settings import SearchSettingsManager

    SearchSettingsManager(tmp_path).set_engine('arxiv')

    class Tool:
        SERVER_NAME = 'web_search'
        _engine_types = ['tavily']
        _api_keys = {}
        _engines = {'tavily': object()}

        def __init__(self):
            self.reloaded = False

        async def reload_from_config(self):
            self.reloaded = True
            self._engine_types = ['arxiv']

    tool = Tool()
    agent = _quiet_agent(tmp_path)
    agent.tool_manager = SimpleNamespace(extra_tools=[tool])
    await apply_live_config(agent, None)
    assert tool.reloaded is True
