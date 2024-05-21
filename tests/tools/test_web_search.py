import os

import json
import pytest
from modelscope_agent.tools.web_search.web_search import WebSearch

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_web_search():
    input_params = """{'query': '2024元旦 哈尔滨 天气'}"""
    web_searcher = WebSearch()
    res = web_searcher.call(input_params)

    assert isinstance(res, str)
    json_res = json.loads(res)
    for item in json_res:
        assert item['link'] or item['sniper']


#
# def test_web_search_agent():
#     responses = [
#         "<|startofthink|>{\"api_name\": \"web_search_utils\", \"parameters\": "
#         "{\"query\": \"2024元旦 哈尔滨 天气\"}}<|endofthink|>", 'summarize'
#     ]
#     llm = MockLLM(responses)
#
#     tools = {'web_search_utils': WebSearch()}
#     prompt_generator = MockPromptGenerator()
#     action_parser = MockOutParser('web_search_utils',
#                                   {'query': '2024元旦 哈尔滨 天气'})
#
#     agent = AgentExecutor(
#         llm,
#         additional_tool_list=tools,
#         prompt_generator=prompt_generator,
#         action_parser=action_parser,
#         tool_retrieval=False,
#     )
#     res = agent.run('帮我查询2024年元旦时哈尔滨天气情况')
#     print(res)
#
#     for item in res[0]['result']:
#         assert item['link'] and item['sniper']
