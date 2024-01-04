import re

from modelscope_agent.agent import AgentExecutor
from modelscope_agent.tools import WebBrowser
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator, MockTool


def test_web_browsing():
    # test code interpreter
    input = 'https://blog.sina.com.cn/zhangwuchang'
    kwargs = {'urls': input}
    web_browser = WebBrowser()
    res = web_browser._local_call(**kwargs)

    assert input == res['result'][0]['url']
    assert len(res['result']) == 1
    assert '张五常' in res['result'][0]['content']


def test_web_browsing_with_split():
    # test code interpreter
    input = ['https://blog.sina.com.cn/zhangwuchang']
    config = {'web_browser': {'split_url_into_chunk': True}}
    kwargs = {'urls': input}
    web_browser = WebBrowser(config)
    res = web_browser._local_call(**kwargs)
    print(res)

    assert input[0] == res['result'][0]['url']
    assert len(res['result']) > 1
    assert '王安石' in res['result'][0]['content']
    assert '张五常' in res['result'][3]['content']


def test_integrated_web_browser_agent():
    llm = MockLLM('')

    tools = {'web_browser': WebBrowser()}
    prompt_generator = MockPromptGenerator()
    url = 'https://blog.sina.com.cn/zhangwuchang'
    action_parser = MockOutParser('web_browser', {'urls': [url]})
    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        action_parser=action_parser,
        tool_retrieval=False,
    )
    res = agent.run('please search the information about zhangwuchang')
    print(res)
    assert url == res[0]['result'][0]['url']
    assert '张五常' in res[0]['result'][0]['content']
