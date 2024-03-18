import pytest
import ray
from modelscope_agent import create_component
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.environment import Environment


@pytest.fixture
def environment():
    roles = ['test_agent1']
    env = create_component(Environment, name='env', roles=roles, remote=False)
    return env


@pytest.fixture
def agent_sender(environment):
    agent_mixin = create_component(
        AgentEnvMixin, name='test_agent2', role='test_agent2', remote=False)
    return agent_mixin


@pytest.fixture
def agent_getter(environment):
    agent_mixin = create_component(
        AgentEnvMixin,
        name='test_agent3',
        role='test_agent3',
        remote=False,
        parse_env_prompt_function=lambda x: x[0].content)
    return agent_mixin


@pytest.fixture
def remote_environments():
    roles = ['test_agent1']
    env = create_component(Environment, name='env', roles=roles, remote=True)
    return env


@pytest.fixture
def remote_agent_sender(environment):
    agent_mixin = create_component(
        AgentEnvMixin, name='test_agent2', role='test_agent2', remote=True)
    return agent_mixin


@pytest.fixture
def remote_agent_getter(environment):
    agent_mixin = create_component(
        AgentEnvMixin,
        name='test_agent3',
        role='test_agent3',
        remote=True,
        parse_env_prompt_function=lambda x: x[0].content)
    return agent_mixin


def test_set_env_context(agent_sender, environment):
    agent_sender.set_env_context(environment)
    assert agent_sender.env_context == environment


def test_role(agent_sender):
    assert agent_sender.role() == 'test_agent2'


def test_publish_pull(agent_sender, agent_getter, environment):
    agent_sender.set_env_context(environment)
    agent_getter.set_env_context(environment)
    environment.register_roles(['test_agent3', 'test_agent2', 'test_agent1'])
    agent_sender.publish('Hello, World!', 'test_agent3')
    message = agent_getter.pull()
    assert message == 'Hello, World!'


def test_publish_pull_all(agent_sender, agent_getter, environment):
    agent_sender.set_env_context(environment)
    agent_getter.set_env_context(environment)
    environment.register_roles(['test_agent3', 'test_agent2', 'test_agent1'])
    agent_sender.publish('Hello, World!')
    message = agent_getter.pull()
    assert message == 'Hello, World!'


def test_publish_pull_remote(remote_agent_sender, remote_agent_getter,
                             remote_environments):
    ray.get(remote_agent_sender.set_env_context.remote(remote_environments))
    ray.get(remote_agent_getter.set_env_context.remote(remote_environments))
    ray.get(
        remote_environments.register_roles.remote(
            ['test_agent3', 'test_agent2', 'test_agent1']))
    ray.get(remote_agent_sender.publish.remote('Hello, World!', 'test_agent3'))
    message = ray.get(remote_agent_getter.pull.remote())
    assert message == 'Hello, World!'
    ray.shutdown()
