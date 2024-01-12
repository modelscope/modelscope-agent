from modelscope_agent.tools import QWenVL

from modelscope_agent.agents.role_play import RolePlay  # NOQA


def test_qwen_vl():
    # 图片默认上传到ci_workspace,后端测试mork时需要在本地存图片到/tmp/ci_workspace，这里只需要图片basename。
    params = """{'image_file_path': 'WechatIMG139.jpg', 'text': '描述这张照片'}"""
    qvl = QWenVL()
    res = qvl.call(params)
    print(res)
    assert (isinstance(res, dict) and 'text' in res)
