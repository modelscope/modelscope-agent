# Copyright (c) ModelScope Contributors. All rights reserved.
"""LLMAgent.set_permission_mode: live mode switch used by the TUI /permission."""
import pytest

from ms_agent.agent.llm_agent import LLMAgent
from ms_agent.permission.config import PermissionConfig


def _agent_with_enforcer():
    agent = LLMAgent.__new__(LLMAgent)
    cfg = PermissionConfig()  # default mode='auto'

    class _Enf:
        pass

    class _TM:
        pass

    enf = _Enf()
    enf._config = cfg
    tm = _TM()
    tm._permission_mode = 'auto'
    tm._permission_config = cfg
    tm._permission_enforcer = enf
    agent.tool_manager = tm
    return agent, tm, enf


def test_switch_updates_toolmanager_and_enforcer():
    agent, tm, enf = _agent_with_enforcer()
    assert agent.set_permission_mode('strict') == 'strict'
    assert tm._permission_mode == 'strict'
    assert tm._permission_config.mode == 'strict'
    assert enf._config.mode == 'strict'


def test_restricted_normalizes_to_interactive():
    agent, tm, enf = _agent_with_enforcer()
    assert agent.set_permission_mode('restricted') == 'interactive'
    assert enf._config.mode == 'interactive'
    assert enf._config.human_approval_available is True


def test_invalid_mode_raises():
    agent, _, _ = _agent_with_enforcer()
    with pytest.raises(ValueError):
        agent.set_permission_mode('bogus')


def test_no_toolmanager_is_safe():
    agent = LLMAgent.__new__(LLMAgent)
    agent.tool_manager = None
    assert agent.set_permission_mode('auto') == 'auto'  # no crash


def test_delegate_mode_and_provider_setter():
    agent, tm, enf = _agent_with_enforcer()
    provider = object()

    assert agent.set_permission_mode('delegate') == 'delegate'
    agent.set_permission_decision_provider(provider)

    assert tm._permission_mode == 'delegate'
    assert enf._config.mode == 'delegate'
    assert enf._provider is provider


def test_full_access_is_supported_as_public_mode():
    agent, tm, enf = _agent_with_enforcer()

    assert agent.set_permission_mode('full_access') == 'full_access'
    assert tm._permission_mode == 'full_access'
    assert enf._config.mode == 'full_access'
    assert PermissionConfig.from_dict({'mode': 'full_access'}).mode == 'full_access'


def test_permission_config_keeps_legacy_positional_order():
    config = PermissionConfig('interactive', ('safe---tool',))

    assert config.mode == 'interactive'
    assert config.whitelist == ('safe---tool',)
