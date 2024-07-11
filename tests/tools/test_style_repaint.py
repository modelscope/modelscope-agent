import os

import pytest
from modelscope_agent.constants import DEFAULT_CODE_INTERPRETER_DIR
from modelscope_agent.tools.dashscope_tools.style_repaint import StyleRepaint
from modelscope_agent.utils.base64_utils import encode_files_to_base64

from modelscope_agent.agents.role_play import RolePlay  # NOQA

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_style_repaint():
    # 图片默认上传到ci_workspace
    params = """{'input.image_path': 'luoli15.jpg', 'input.style_index': 0}"""

    style_repaint = StyleRepaint()
    res = style_repaint.call(params)
    assert (res.startswith('![IMAGEGEN](http'))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_base64_qwen_vl():
    work_dir = os.getenv('CODE_INTERPRETER_WORK_DIR',
                         DEFAULT_CODE_INTERPRETER_DIR)
    params = """{'input.image_path': 'girl.png', 'input.style_index': 1}"""

    file_paths = 'girl.png'.split(',')
    for i, file_path in enumerate(file_paths):
        file_path = os.path.join(work_dir, file_path)
        file_paths[i] = file_path

    base64_files = encode_files_to_base64(file_paths)

    style_repainter = StyleRepaint()
    kwargs = {}
    kwargs['base64_files'] = base64_files
    res = style_repainter.call(params, **kwargs)
    print(res)
    assert (res.startswith('![IMAGEGEN](https://'))


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_style_repaint_role():
    role_template = '你扮演一个绘画家，用尽可能丰富的描述调用工具绘制各种风格的图画。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['style_repaint']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('[上传文件 "luoli15.jpg"],我想要清雅国风')
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
