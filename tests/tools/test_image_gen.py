from modelscope_agent.agent import Agent
from modelscope_agent.tools.image_generation import TextToImageTool

from modelscope_agent.agents.role_play import RolePlay  # NOQA


def test_image_gen():
    params = """{'text': '画一只小猫', 'resolution': '1024*1024'}"""

    t2i = TextToImageTool()
    res = t2i.call(params)
    assert (res.startswith('http'))


def test_image_gen_wrong_resolution():
    params = """{'text': '画一只小猫', 'resolution': '1024'}"""

    t2i = TextToImageTool()
    res = t2i.call(params)
    assert (res.startswith('http'))


def test_image_gen_role():
    role_template = '你扮演一个画家，用尽可能丰富的描述调用工具绘制图像。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = [{'name': 'image_gen'}]

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('朝阳区天气怎样？')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
