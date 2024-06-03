import os

import pytest
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.tools.dashscope_tools.image_generation_lite import \
    TextToImageLiteTool


@pytest.mark.skip(reason='only run with authorized keys')
def test_image_gen_lite():
    params = """
    {'text': '一只可爱的小兔子正在花园里努力地拔一个大萝卜，周围是绿油油的草地和鲜艳的花朵，天空是清澈的蓝色，太阳公公笑眯眯地看着。',
    'lora_index': 'wanxlite1.4.5_lora_huibenlite1_20240519',
    'resolution': '1024*1024'
    }
    """
    t2i = TextToImageLiteTool()
    res = t2i.call(params)
    assert (res.startswith('![IMAGEGEN]('))


@pytest.mark.skip(reason='only run with authorized keys')
def test_image_gen_lite_role():
    role_template = '扮演一个绘本小助手，可以利用工具来创建符合儿童的童话绘本图片'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['image_gen_lite']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('绘制一个小兔子拔萝卜的场景，使用lora来控制风格')
    text = ''
    for chunk in response:
        text += chunk
    assert isinstance(text, str)
