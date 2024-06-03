import os

import pytest
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.tools.dashscope_tools.image_generation import \
    TextToImageTool

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_image_gen():
    params = """{'text': '画一只小猫', 'resolution': '1024*1024'}"""

    t2i = TextToImageTool()
    res = t2i.call(params)
    assert (res.startswith('![IMAGEGEN]('))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_image_gen_wrong_resolution():
    params = """{'text': '画一只小猫', 'resolution': '1024'}"""

    t2i = TextToImageTool()
    res = t2i.call(params)
    assert (res.startswith('![IMAGEGEN]('))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_image_gen_with_lora():
    params = """{'text': '画一只小猫', 'resolution': '1024*1024', 'lora_index': 'wanx1.4.5_textlora_huiben2_20240518'}"""
    t2i = TextToImageTool()
    res = t2i.call(params)
    assert (res.startswith('![IMAGEGEN]('))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_image_gen_role():
    role_template = '你扮演一个画家，用尽可能丰富的描述调用工具绘制图像。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['image_gen']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('画一张猫的图像')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
