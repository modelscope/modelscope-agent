from modelscope_agent.agents.role_play import RolePlay  # NOQA

import os

import pytest

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_weather_role():
    role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool name
    function_list = ['amap_weather']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('朝阳区天气怎样？')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
