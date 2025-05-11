from modelscope_agent.agent import Agent  # NOQA

import os

def test_web_gen():

    # input tool name
    mcp_servers = {
        "mcpServers": {
            "time": {
                "type": "sse",
                "url": "https://agenttor-mod-dd-cbwtrtihpn.cn-zhangjiakou.fcapp.run/sse"
            },
            "fetch": {
                "type":
                    "sse",
                "url":
                    "https://mcp-cdb79f47-15a7-4a72.api-inference.modelscope.cn/sse"
            }
        }
    }

    default_system = ('You are an assistant which helps me to finish a complex job. Tools may be given to you '
                      'and you must choose some of them one per round to finish my request.')

    # TODO: compatible with qwen_agent
    llm_config = {
        'model': 'Qwen/Qwen2.5-72B-Instruct',
        'model_server': 'https://api-inference.modelscope.cn/v1/',   # https://api.openai.com/v1/chat/completions
        'api_key': os.getenv('MODELSCOPE_API_KEY')
    }
    # llm_config = {
    #     'model': 'Qwen/Qwen2.5-72B-Instruct',
    #     'model_server': 'openai',
    #     'api_base': 'https://api-inference.modelscope.cn/v1/',
    #     'api_key': os.getenv('MODELSCOPE_API_KEY')
    # }
    # llm_config = {
    #     'model': 'claude-3-7-sonnet-20250219',
    #     'model_server': 'openai',
    #     'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    #     'api_key': os.getenv('DASHSCOPE_API_KEY_YH')
    # }

    bot = Agent(
        mcp=mcp_servers, llm=llm_config, instruction=default_system)

    # Construct the request
    messages = [
        {
            'role': 'system',
            'content': default_system
        },
        {
            'role': 'user',
            'content': '上周日几号？那天北京天气情况如何'
        }
    ]

    response = bot.run(
        messages=messages,
        stream=True,
        seed=None,
    )

    for res in response:
        print(res)

    # text = ''
    # for chunk in response:
    #     text += chunk
    # print(text)
    # assert isinstance(text, str)


if __name__ == '__main__':

    test_web_gen()
