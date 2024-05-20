from modelscope_agent.tools.web_browser import WebBrowser


def test_web_browser():
    # test web browsing
    params = """{'urls': 'https://blog.sina.com.cn/zhangwuchang'}"""
    web_browser = WebBrowser()
    res = web_browser.call(params=params)

    assert isinstance(res, str)
    assert len(res) == 2000
    assert '张五常' in res


def test_web_browser_with_length():
    # test web browsing
    params = """{'urls': 'https://blog.sina.com.cn/zhangwuchang'}"""
    web_browser = WebBrowser()
    res = web_browser.call(params=params, max_browser_length=100)

    assert isinstance(res, str)
    assert len(res) == 100


def test_web_browser_with_adv():
    params = """{'urls': ['https://www.bing.com/search?q=拜登当选总统的年份是哪一年']}"""
    tool_config = {'web_browser': {'use_adv': True}}
    web_browser = WebBrowser(tool_config)
    res = web_browser.call(params=params)
    assert isinstance(res, str)


# def test_integrated_web_browser_agent():
#     llm = MockLLM('')
#
#     tools = {'web_browser': WebBrowser()}
#     prompt_generator = MockPromptGenerator()
#     url = 'https://blog.sina.com.cn/zhangwuchang'
#     action_parser = MockOutParser('web_browser', {'urls': [url]})
#     agent = AgentExecutor(
#         llm,
#         additional_tool_list=tools,
#         prompt_generator=prompt_generator,
#         action_parser=action_parser,
#         tool_retrieval=False,
#     )
#     res = agent.run('please search the information about zhangwuchang')
#     print(res)
#     assert url == res[0]['result'][0]['url']
#     assert '张五常' in res[0]['result'][0]['content']
