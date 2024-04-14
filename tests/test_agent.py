import pytest
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.llm import BaseChatModel
from modelscope_agent.tools import TOOL_REGISTRY, BaseTool

from .ut_utils import MockTool


class MockTool1(MockTool):
    name: str = 'mock_tool1'


# Using RolePlay as a concrete agent
@pytest.fixture
def tester_agent(mocker):
    TOOL_REGISTRY['mock_tool'] = MockTool
    TOOL_REGISTRY['mock_tool1'] = MockTool1
    function_list = ['mock_tool', {'mock_tool1': {'config': 'some_config'}}]
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'test'
    }
    agent = RolePlay(
        function_list=function_list,
        llm=llm_config,
        storage_path='/path/to/storage',
        name='ConcreteAgent',
        description='A concrete agent for testing',
        instruction='Role player for testing')
    mocker.patch.object(agent, '_run', return_value=['hello', ' there'])
    return agent


def test_agent_initialization(tester_agent):
    assert isinstance(tester_agent.function_list, list)
    assert len(tester_agent.function_list) == 2
    assert 'mock_tool1' in tester_agent.function_map
    assert 'mock_tool' in tester_agent.function_map

    assert tester_agent.llm is not None  # Assuming the llm was provided as a config dict
    assert isinstance(
        tester_agent.llm,
        BaseChatModel)  # Ensure llm is an instance of BaseChatModel

    assert tester_agent.storage_path == '/path/to/storage'
    assert tester_agent.name == 'ConcreteAgent'
    assert tester_agent.description == 'A concrete agent for testing'
    assert tester_agent.instruction == 'Role player for testing'


def test_agent_run(tester_agent):
    responses = list(tester_agent.run('Hello, world!'))
    assert isinstance(responses, list)
    assert all(isinstance(resp, str) for resp in responses)
    assert all(resp.strip() for resp in responses)
    assert responses == ['hello', ' there']


def test_agent_call_tool(tester_agent):
    # Mocking a simple response from the tool for testing purposes
    response = tester_agent._call_tool('mock_tool', 'tool response')
    assert response == 'tool response'
