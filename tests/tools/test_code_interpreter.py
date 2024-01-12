import re

import json
from modelscope_agent.tools.code_interpreter import CodeInterpreter


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
