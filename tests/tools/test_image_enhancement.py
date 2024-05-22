from modelscope_agent.agents.role_play import RolePlay  # NOQA
import os

import json
import pytest
from modelscope_agent.tools.dashscope_tools.image_enhancement import \
    ImageEnhancement

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_image_enhancement():
    image_url = 'luoli15.jpg'
    kwargs = {'input.image_path': image_url, 'parameters.upscale': 2}
    phantom = ImageEnhancement()
    res = phantom.call(json.dumps(kwargs))
    assert (res.startswith('![IMAGEGEN](http'))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_image_enhancement_agent():
    role_template = '你扮演一个绘画家，用尽可能丰富的描述调用工具绘制各种风格的图画。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['image_enhancement']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('[上传文件 "luoli15.jpg"], 2倍超分这张图')
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
