from modelscope_agent.tools import StyleRepaint

from modelscope_agent.agents.role_play import RolePlay  # NOQA


def test_style_repaint():
    # 图片默认上传到ci_workspace
    params = """{'input.image_path': 'luoli15.jpg', 'input.style_index': 0}"""

    style_repaint = StyleRepaint()
    res = style_repaint.call(params)
    assert (res.startswith('![IMAGEGEN](http'))


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
