"""TUI host defaults for permission escalation and short shell wait."""
from omegaconf import OmegaConf

from ms_agent.tui.app import TuiApp


def test_tui_prepare_config_sets_human_flag_and_short_timeout(tmp_path):
    config = OmegaConf.create({'llm': {'model': 'x'}})
    prepared = TuiApp._prepare_config(config, 'delegate', str(tmp_path))
    assert prepared.permission.human_approval_available is True
    assert prepared.permission.mode == 'delegate'
    assert prepared.tool_call_timeout == 15


def test_tui_prepare_config_keeps_explicit_tool_timeout(tmp_path):
    config = OmegaConf.create({'tool_call_timeout': 45})
    prepared = TuiApp._prepare_config(config, None, str(tmp_path))
    assert prepared.tool_call_timeout == 45
    assert prepared.permission.human_approval_available is True


def test_default_agent_yaml_task_control_is_builtin_not_mcp():
    """task_control must not be treated as an MCP server (no url/command)."""
    from pathlib import Path

    from ms_agent.config.config import Config

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / 'ms_agent' / 'agent' / 'agent.yaml')
    mcp = Config.convert_mcp_servers_to_json(cfg)
    assert 'task_control' not in mcp.get('mcpServers', {})
    assert cfg.tools.task_control.mcp is False


def test_default_yaml_keeps_real_mcp_servers_and_builtin_task_control():
    """Builtins stay extra_tools; a real MCP entry still lands in mcpServers."""
    from pathlib import Path

    from ms_agent.config.config import Config
    from ms_agent.config.mcp_schema import collect_builtin_tool_names
    from ms_agent.tools.mcp_client import MCPClient
    from ms_agent.tools.task_control_tool import TaskControlTool
    from ms_agent.tools.tool_manager import ToolManager

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / 'ms_agent' / 'agent' / 'agent.yaml')
    OmegaConf.update(
        cfg,
        'tools.fetch',
        {
            'mcp': True,
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-fetch'],
        },
        merge=True,
    )

    from_yaml = Config.convert_mcp_servers_to_json(cfg)['mcpServers']
    assert 'fetch' in from_yaml
    assert from_yaml['fetch']['command'] == 'npx'
    for builtin in ('task_control', 'file_system', 'code_executor'):
        assert builtin not in from_yaml

    assert {'task_control', 'file_system', 'code_executor'} <= (
        collect_builtin_tool_names(cfg))

    extra = {
        'mcpServers': {
            'time': {
                'type': 'sse',
                'url': 'http://127.0.0.1:9/sse',
            }
        }
    }
    client = MCPClient(extra, cfg)
    servers = client.mcp_config['mcpServers']
    assert 'fetch' in servers
    assert 'time' in servers
    assert 'task_control' not in servers

    tm = ToolManager(cfg, mcp_config=extra)
    assert any(isinstance(t, TaskControlTool) for t in tm.extra_tools)
