from modelscope_agent.agent import AgentExecutor
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator, MockTool


def _get_llm(responses=['mock_llm_response']):
    llm = MockLLM(responses)
    return llm


def _get_tool(name='search_tool', func=lambda x: x, description='mock tool'):
    tool = MockTool(name, func, description)
    return tool


def _get_output_parser(action, args):
    output_parser = MockOutParser(action, args)
    return output_parser


def _get_agent(llm, tools={}, prompt_generator=None, output_parser=None):
    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        output_parser=output_parser)
    return agent


def test_agent_run():

    responses = [
        "<|startofthink|>{\"api_name\": \"search_tool\", \"parameters\": {\"x\": \"mock task\"}}<|endofthink|>",
        'summaerize'
    ]
    llm = _get_llm(responses)

    def func_tool(x, **kwargs):
        return x

    tools = {'search_tool': _get_tool('search_tool', func_tool)}
    agent = _get_agent(llm, tools)
    res = agent.run('mock task')
    assert res == [{'result': 'mock task'}]


def test_unknown_action_error():
    # test when llm and parser return unknown action
    llm = _get_llm()
    tools = {'search_tool': _get_tool('search_tool')}
    output_parser = _get_output_parser('fake_search_tool',
                                       {'query': 'mock query'})
    prompt_generator = MockPromptGenerator()
    agent = _get_agent(
        llm,
        tools,
        prompt_generator=prompt_generator,
        output_parser=output_parser)
    res = agent.run('mock task')
    assert res == [{'error': "Unknown action: 'fake_search_tool'. "}]


def test_tool_execute_error():
    # test when action calling error
    llm = _get_llm()
    tools = {
        'search_tool': _get_tool('search_tool', lambda x: 1 / 0, 'mock tool')
    }
    output_parser = _get_output_parser('search_tool', {'query': 'mock query'})
    prompt_generator = MockPromptGenerator()
    agent = _get_agent(
        llm,
        tools,
        prompt_generator=prompt_generator,
        output_parser=output_parser)
    res = agent.run('mock task')
    assert res[0]['error'].startswith(
        "Action call error: search_tool: {'query': 'mock query'}.")
