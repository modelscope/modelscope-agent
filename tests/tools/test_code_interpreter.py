from modelscope_agent.agent import AgentExecutor
from modelscope_agent.tools import CodeInterpreter
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator, MockTool


def test_code_interpreter_python():
    # test code interpreter
    kwargs = {'language': 'python', 'code': 'print(1)'}
    code_interpreter = CodeInterpreter()
    res = code_interpreter._local_call(**kwargs)

    assert res['result'] == '1'


def test_code_interpreter_shell():
    # test code interpreter
    kwargs = {'language': 'shell', 'code': 'echo 1'}
    code_interpreter = CodeInterpreter()
    res = code_interpreter._local_call(**kwargs)

    assert res['result'] == '1'


def test_integrated_code_interpreter_agent():
    responses = [
        "<|startofthink|>{\"api_name\": \"code_interpreter\", \"parameters\": "
        "{\"language\": \"python\", \"code\": \"print(1)\"}}<|endofthink|>",
        'summarize'
    ]
    llm = MockLLM(responses)

    tools = {'code_interpreter': CodeInterpreter()}
    prompt_generator = MockPromptGenerator()
    output_parser = MockOutParser('code_interpreter', {
        'language': 'python',
        'code': 'print(1)'
    })
    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        output_parser=output_parser,
        tool_retrieval=False,
    )
    res = agent.run('please generate code to print 1 in python')
    assert res == [{'result': '1'}]
