import os

import pytest
from modelscope_agent.tools.dashscope_tools.wordart_tool import WordArtTexture

from modelscope_agent.agents.role_play import RolePlay  # NOQA

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_word_art():
    params = """{
        'input.text.text_content': '魔搭社区',
        'input.prompt': '一片绿色的森林里开着小花',
        'input.texture_style': 'scene',
        'input.text.output_image_ratio': '9:16'
    }"""
    wa = WordArtTexture()
    res = wa.call(params)
    print(res)
    assert (res.startswith('![IMAGEGEN](https://'))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_word_art_role():
    role_template = '你扮演一个美术老师，用尽可能丰富的描述调用工具生成艺术字图片。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['wordart_texture_generation']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('文字内容：你好新年,风格：海洋，纹理风格：默认，宽高比：16:9')
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
