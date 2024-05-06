from modelscope_agent.tools import QWenVL

from modelscope_agent.agents.role_play.role_play import RolePlay  # NOQA


def test_qwen_vl():
    # 图片默认上传到ci_workspace,后端测试mork时需要在本地存图片到/tmp/ci_workspace，这里只需要图片basename。
    params = """{'image_file_path': 'luoli15.jpg', 'text': '描述这张照片'}"""
    qvl = QWenVL()
    res = qvl.call(params)
    print(res)
    assert (isinstance(res, dict) and 'text' in res)


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
