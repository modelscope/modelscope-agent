from modelscope_agent.agent import AgentExecutor
from modelscope_agent.tools import WebSearch
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator, MockTool


def test_web_search():
    input = '2024元旦 哈尔滨 天气'
    kwargs = {'query': input}
    web_searcher = WebSearch()
    res = web_searcher._remote_call(**kwargs)

    for item in res['result']:
        assert item['link'] and item['sniper']


def test_web_search_agent():
    responses = [
        "<|startofthink|>{\"api_name\": \"web_search\", \"parameters\": "
        "{\"query\": \"2024元旦 哈尔滨 天气\"}}<|endofthink|>", 'summarize'
    ]
    llm = MockLLM(responses)

    tools = {'web_search': WebSearch()}
    prompt_generator = MockPromptGenerator()
    output_parser = MockOutParser('web_search', {'query': '2024元旦 哈尔滨 天气'})

    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        output_parser=output_parser,
        tool_retrieval=False,
    )
    res = agent.run('帮我查询2024年元旦时哈尔滨天气情况')
    print(res)

    for item in res[0]['result']:
        assert item['link'] and item['sniper']
