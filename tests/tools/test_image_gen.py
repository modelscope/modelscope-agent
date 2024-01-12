from modelscope_agent.tools.dashscope_tools import TextToImageTool


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
