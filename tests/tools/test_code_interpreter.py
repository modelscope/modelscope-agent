import re

import json
from modelscope_agent.tools.code_interpreter.code_interpreter import \
    CodeInterpreter


def test_code_interpreter_image():
    # test code interpreter
    input = """import matplotlib.pyplot as plt\nimport numpy as np\n\n# 设置参数范围\nt = np.linspace(0, 2 * np.pi, 1000)\n\n# 定义心形图的方程\nx = 16 * np.sin(t)**3\ny = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)\n\n# 创建图形\nplt.figure(figsize=(8, 6))\n\n# 绘制心形图\nplt.plot(x, y, color='red')\n\n# 设置坐标轴比例\nplt.axis('equal')\n\n# 隐藏坐标轴\nplt.axis('off')\n\n# 显示图形\nplt.show()"""  # noqa E501
    kwargs = {'code': input}
    code_interpreter = CodeInterpreter()
    res = code_interpreter.call(json.dumps(kwargs))

    assert '![IMAGEGEN]' in res
    re_pattern1 = re.compile(pattern=r'!\[IMAGEGEN\]\(([\s\S]+)\)')

    res = re_pattern1.search(res)
    # decide if file is .png file
    assert res.group(1).endswith('.png')


def test_code_interpreter_jupyter_text():
    # test code interpreter
    input = 'print(1)'
    kwargs = {'code': input}
    code_interpreter = CodeInterpreter()
    res = code_interpreter.call(json.dumps(kwargs))

    assert res.strip() == '1'


def test_code_interpreter_video():
    # test code interpreter
    input = """from moviepy.editor import ImageClip, concatenate_videoclips\n\n# 读取图片\nimage = ImageClip("luoli15.jpg")\n\n# 创建一个持续5秒的循环动画\nanimation = image.set_duration(5).rotate(angle=360)\n\n# 将动画保存为mp4文件\nanimation.write_videofile("rotating_image.mp4", fps=24)\n\n"""  # noqa E501

    kwargs = {'code': input}
    code_interpreter = CodeInterpreter()
    res = code_interpreter.call(json.dumps(kwargs))

    assert '<audio src="' in res

    re_pattern1 = re.compile(pattern=r'([\s\S]+)<audio src="([\s\S]+)"/>')
    res = re_pattern1.search(res)
    # decide if file is .mp4 file
    assert res.group(2).endswith('.mp4')


def test_code_interpreter_nd_mode():
    input_1 = """a=1+2\na"""
    input_2 = """a=a*2\na"""
    code_interpreter = CodeInterpreter()
    kwargs_1 = {'code': input_1, 'nb_mode': True}
    kwargs_2 = {'code': input_2, 'nb_mode': True}
    res_1 = code_interpreter.call(json.dumps(kwargs_1))
    res_2 = code_interpreter.call(json.dumps(kwargs_2))
    assert res_1 == '3'
    assert res_2 == '6'
