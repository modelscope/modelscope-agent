from modelscope_agent.agent import Agent  # NOQA

import os

import pytest

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_weather_role():
    llm_config = {
        'model': 'Qwen/Qwen2.5-72B-Instruct',
        'model_server': 'openai',
        'api_base': 'https://api-inference.modelscope.cn/v1/',
        'api_key': os.getenv('MODELSCOPE_API_KEY')
    }

    # input tool name
    mcp_servers = {
        'mcpServers': {
            'time': {
                'type':
                'sse',
                'url':
                'https://agenttor-mod-dd-cbwtrtihpn.cn-zhangjiakou.fcapp.run/sse'
            },
            'fetch': {
                'type':
                'sse',
                'url':
                'https://mcp-cdb79f47-15a7-4a72.api-inference.modelscope.cn/sse'
            }
        }
    }

    default_system = (
        'You are an assistant which helps me to finish a complex job. Tools may be given to you '
        'and you must choose some of them one per round to finish my request.')
    bot = Agent(mcp=mcp_servers, llm=llm_config, instruction=default_system)

    response = bot.run('今天是哪一天？今天热门人工智能新闻有哪些？')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)


test_weather_role()
