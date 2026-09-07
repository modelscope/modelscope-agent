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
