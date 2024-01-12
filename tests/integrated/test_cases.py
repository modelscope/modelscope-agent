from modelscope_agent.agents.role_play import RolePlay

llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}


def test_image_gen_role():
    role_template = '你扮演一个画家，用尽可能丰富的描述调用工具绘制图像。'

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


def test_weather_role():
    role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

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
