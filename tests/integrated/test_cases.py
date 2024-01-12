from modelscope_agent.agents.role_play import RolePlay


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


def test_weather_role():
    role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

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


def test_style_repaint_role():
    role_template = '你扮演一个绘画家，用尽可能丰富的描述调用工具绘制各种风格的图画。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['style_repaint']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('[上传文件WechatIMG139.jpg],我想要清雅国风')
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)


def test_qwen_vl_role():
    role_template = '你扮演一个美术老师，用尽可能丰富的描述调用工具讲解描述各种图画。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = ['qwen_vl']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('[上传文件WechatIMG139.jpg],描述这张照片')
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
