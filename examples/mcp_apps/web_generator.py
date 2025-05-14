import os

from modelscope_agent.agents.agent_with_mcp import AgentWithMCP  # NOQA


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

    llm_config = {
        'model':
        'Qwen/Qwen3-235B-A22B',  # Qwen/Qwen3-235B-A22B, Qwen/Qwen2.5-72B-Instruct
        'model_server': 'https://api-inference.modelscope.cn/v1/',
        'api_key': os.getenv('MODELSCOPE_API_KEY')
    }

    # ONLY FOR TEST
    function_list = [{
        'mcpServers': {
            'amap-maps': {
                'type': 'sse',
                'url': 'https://mcp.api-inference.modelscope.cn/sse/xxx'
            },
            'MiniMax-MCP': {
                'type': 'sse',
                'url': 'https://mcp.api-inference.modelscope.cn/sse/xxx'
            },
            'edgeone-pages-mcp-server': {
                'command': 'npx',
                'args': ['edgeone-pages-mcp']
            },
            'notebook': None,
        }
    }]

    bot = AgentWithMCP(
        function_list=function_list,
        mcp=mcp_servers,
        llm=llm_config,
    )

    # Construct requests
    messages = [{
        'role': 'user',
        # 'content': '上周日几号？那天北京天气情况如何'
        'content': '做一个图文并茂的绘本故事，部署成一个网页。'
    }]

    kwargs = {}
    kwargs.update({
        'stream': True,
        'max_tokens': 16384,
        'extra_body': {
            'enable_thinking': False
        }
    })

    response = bot.run(
        messages=messages,
        **kwargs,
    )

    for chunk in response:
        print('\n')
        print('-' * 100)
        print(f'>>chunk: {chunk}')
        print('+' * 100)
        print('\n')


if __name__ == '__main__':

    test_web_gen()
