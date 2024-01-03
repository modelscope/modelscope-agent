import re

from agent_scope.agent import AgentExecutor
from agent_scope.tools import CodeInterpreter, CodeInterpreterJupyter
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator, MockTool


def test_code_interpreter_jupyter_image():
    # test code interpreter
    input = """import matplotlib.pyplot as plt\nimport numpy as np\n\n# 设置参数范围\nt = np.linspace(0, 2 * np.pi, 1000)\n\n# 定义心形图的方程\nx = 16 * np.sin(t)**3\ny = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)\n\n# 创建图形\nplt.figure(figsize=(8, 6))\n\n# 绘制心形图\nplt.plot(x, y, color='red')\n\n# 设置坐标轴比例\nplt.axis('equal')\n\n# 隐藏坐标轴\nplt.axis('off')\n\n# 显示图形\nplt.show()"""  # noqa E501
    kwargs = {'code': input}
    code_interpreter = CodeInterpreterJupyter()
    res = code_interpreter._local_call(**kwargs)

    assert '![IMAGEGEN]' in res['result']
    re_pattern1 = re.compile(pattern=r'!\[IMAGEGEN\]\(([\s\S]+)\)')

    res = re_pattern1.search(res['result'])
    # decide if file is .png file
    assert res.group(1).endswith('.png')


def test_code_interpreter_jupyter_text():
    # test code interpreter
    input = 'print(1)'
    kwargs = {'code': input}
    code_interpreter = CodeInterpreterJupyter()
    res = code_interpreter._local_call(**kwargs)

    assert res['result'].strip() == '1'


def test_integrated_code_interpreter_agent():
    responses = [
        "<|startofthink|>{\"api_name\": \"code_interpreter\", \"parameters\": "
        "{\"language\": \"python\", \"code\": \"print(1)\"}}<|endofthink|>",
        'summarize'
    ]
    llm = MockLLM(responses)

    tools = {'code_interpreter': CodeInterpreterJupyter()}
    prompt_generator = MockPromptGenerator()
    output_parser = MockOutParser('code_interpreter', {'code': 'print(1)'})
    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        output_parser=output_parser,
        tool_retrieval=False,
    )
    res = agent.run('please generate code to print 1 in python')
    print(res)
    assert res[0]['result'].strip() == '1'
