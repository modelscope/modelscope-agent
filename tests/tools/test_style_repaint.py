from modelscope_agent.tools import StyleRepaint

from modelscope_agent.agents.role_play import RolePlay  # NOQA


def test_style_repaint():
    # 图片默认上传到ci_workspace
    params = """{'input.image_path': './WechatIMG139.jpg', 'input.style_index': 0}"""

    style_repaint = StyleRepaint()
    res = style_repaint.call(params)
    assert (res.startswith('http'))
