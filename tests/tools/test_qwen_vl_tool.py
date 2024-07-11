import os

import pytest
from modelscope_agent.constants import DEFAULT_CODE_INTERPRETER_DIR
from modelscope_agent.tools.dashscope_tools.qwen_vl import QWenVL
from modelscope_agent.utils.base64_utils import encode_files_to_base64

from modelscope_agent.agents.role_play import RolePlay  # NOQA

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_single_qwen_vl():
    # 图片默认上传到ci_workspace,后端测试mork时需要在本地存图片到/tmp/ci_workspace，这里只需要图片basename。
    params = """{'image_file_paths': 'luoli15.jpg', 'text': '描述这张照片'}"""
    qvl = QWenVL()
    res = qvl.call(params)
    print(res)
    assert (isinstance(res, dict) and 'text' in res)


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_multi_qwen_vl():
    # 图片默认上传到ci_workspace,后端测试mork时需要在本地存图片到/tmp/ci_workspace，这里只需要图片basename。
    params = """{'image_file_paths': 'luoli15.jpg,girl.png', 'text': '比较这两张的不同'}"""
    qvl = QWenVL()
    res = qvl.call(params)
    print(res)
    assert (isinstance(res, dict) and 'text' in res)


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_base64_qwen_vl():
    work_dir = os.getenv('CODE_INTERPRETER_WORK_DIR',
                         DEFAULT_CODE_INTERPRETER_DIR)
    params = """{'image_file_paths': 'luoli15.jpg,girl.png', 'text': '比较这两张的不同'}"""

    file_paths = 'luoli15.jpg,girl.png'.split(',')
    for i, file_path in enumerate(file_paths):
        file_path = os.path.join(work_dir, file_path)
        file_paths[i] = file_path

    base64_files = encode_files_to_base64(file_paths)

    qvl = QWenVL()
    kwargs = {}
    kwargs['base64_files'] = base64_files
    res = qvl.call(params, **kwargs)
    print(res)
    assert (isinstance(res, dict) and 'text' in res)


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_qwen_vl_role():
    role_template = '你扮演一个美术老师，用尽可能丰富的描述调用工具讲解描述各种图画。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['qwen_vl']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('[上传文件luoli15.jpg],描述这张照片')
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
