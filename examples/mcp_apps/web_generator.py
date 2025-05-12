from modelscope_agent.agents.agent_with_mcp import AgentWithMCP  # NOQA

import os


def test_web_gen():

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

    llm_config = {
        'model': 'Qwen/Qwen2.5-72B-Instruct',
        'model_server': 'https://api-inference.modelscope.cn/v1/',
        'api_key': os.getenv('MODELSCOPE_API_KEY')
    }

    bot = AgentWithMCP(
        mcp=mcp_servers, llm=llm_config, instruction=default_system)

    # Construct requests
    messages = [{
        'role': 'system',
        'content': default_system
    }, {
        'role': 'user',
        'content': '上周日几号？那天北京天气情况如何'
    }]

    response = bot.run(
        messages=messages,
        stream=True,
        seed=None,
    )

    for chunk in response:
        print(chunk)


if __name__ == '__main__':

    test_web_gen()
