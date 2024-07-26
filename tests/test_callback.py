import os

import pytest
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.callbacks.run_state import RunStateCallback
from modelscope_agent.memory import MemoryWithRag
from modelscope_agent.tools.base import TOOL_REGISTRY

from .ut_utils import MockTool

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_llm_run_state(mocker):
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'test'
    }

    callback = RunStateCallback()

    agent = RolePlay(llm=llm_config, callbacks=[callback], stream=False)

    mocker.patch.object(agent.llm, '_chat_no_stream', return_value='hello')
    response = agent.run('hello')
    for r in response:
        print(r)

    assert callback.run_states[1][0].type == 'llm'
    assert callback.run_states[1][0].content == 'hello'


def test_tool_exec_run_state(mocker):
    TOOL_REGISTRY['mock_tool'] = {'class': MockTool}
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'test'
    }
    function_list = ['mock_tool']

    callback = RunStateCallback()

    agent = RolePlay(
        llm=llm_config,
        callbacks=[callback],
        function_list=function_list,
        stream=False)

    mocker.patch.object(
        agent.llm,
        '_chat_no_stream',
        return_value='Action: mock_tool\nAction Input: {"test": "test_value"}')

    response = agent.run('hello')

    for r in response:
        print(r)

    assert callback.run_states[1][1].type == 'tool_input'
    assert callback.run_states[1][1].name == 'mock_tool'
    assert callback.run_states[1][1].content == '{"test": "test_value"}'
    assert callback.run_states[1][2].type == 'tool_output'
    assert callback.run_states[1][2].name == 'mock_tool'
    assert callback.run_states[1][2].content == '{"test": "test_value"}'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_rag_run_state(mocker):

    callback = RunStateCallback()
    memory = MemoryWithRag(
        urls=['tests/samples/常见QA.pdf'],
        use_knowledge_cache=False,
        callbacks=[callback])

    memory.run(query='高德天气api怎么申请')
    assert callback.run_states[1][0].type == 'rag'
