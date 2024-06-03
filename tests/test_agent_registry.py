from unittest.mock import MagicMock

import pytest
from modelscope_agent import create_component
from modelscope_agent.agents import RolePlay
from modelscope_agent.agents_registry import \
    AgentRegistry  # Adjust the import path as needed
from modelscope_agent.environment import Environment


@pytest.fixture
def agent_registry():
    return AgentRegistry(remote=False)


@pytest.fixture
def mock_agent(mocker):
    # using mock llm result as output
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'test'
    }
    test_agent = create_component(
        RolePlay,
        name='test_agent',
        remote=False,
        llm=llm_config,
        function_list=[],
        role='test_agent',
    )
    mocker.patch.object(test_agent, '_run', return_value=['hello', ' there'])
    return test_agent


@pytest.fixture
def mock_env(mocker):
    env = MagicMock(spec=Environment)
    return env


def test_register_agent(agent_registry, mock_agent, mock_env):
    agent_registry.register_agent(mock_agent, env_context=mock_env)
    assert 'test_agent' in agent_registry._agents
    assert agent_registry._agents['test_agent'] == mock_agent
    assert agent_registry._agents_state['test_agent'] is True


def test_get_agents_by_role(agent_registry, mock_agent, mock_env):
    agent_registry.register_agent(mock_agent, env_context=mock_env)
    agent = agent_registry.get_agent_by_role('test_agent')
    assert agent == mock_agent


def test_register_agents(agent_registry, mock_agent, mock_env):
    agents = [mock_agent]
    agent_registry.register_agents(agents, env_context=mock_env)
    roles = ['test_agent']
    agents = agent_registry.get_agents_by_role(roles)
    assert agents['test_agent'] == mock_agent


# More test cases can be added for other methods

# To run the tests use:
# pytest test_agent_registry.py
