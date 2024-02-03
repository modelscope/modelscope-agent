from modelscope_agent.agent import AgentExecutor
from modelscope_agent.tools import Phantom
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator, MockTool


def test_phantom():
    input = '2_local_user.png'
    kwargs = {
        'input.image_path': input,
        'parameters.upscale': 2,
        'remote': False
    }
    phantom = Phantom()
    res = phantom(**kwargs)

    print(res)
    assert res['result']['url'].startswith('http')


def test_phantom_agent():
    responses = [
        "<|startofthink|>{\"api_name\": \"phantom_image_enhancement\", \"parameters\": "
        "{\"input.image_path\": \"2_local_user.png\"}}<|endofthink|>",
        'summarize'
    ]
    llm = MockLLM(responses)

    tools = {'phantom_image_enhancement': Phantom()}
    prompt_generator = MockPromptGenerator()
    action_parser = MockOutParser('phantom_image_enhancement',
                                  {'input.image_path': '2_local_user.png'})

    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        action_parser=action_parser,
        tool_retrieval=False,
    )
    res = agent.run('2倍超分')
    print(res)

    assert res[0]['result']['url'].startswith('http')
