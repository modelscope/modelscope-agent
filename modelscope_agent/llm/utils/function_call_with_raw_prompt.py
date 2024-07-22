import re
from typing import List, Tuple, Union

DEFAULT_EXEC_TEMPLATE = """\nObservation: <result>{exec_result}</result>\nAnswer:"""

ACTION_TOKEN = 'Action:'
ARGS_TOKEN = 'Action Input:'
OBSERVATION_TOKEN = 'Observation:'
ANSWER_TOKEN = 'Answer:'

TOOL_TEMPLATE_ZH = """
# 工具

## 你拥有如下工具：

{tool_descs}

## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令，可以根据需求调用零次或多次：

工具调用
Action: 工具的名称，必须是[{tool_names}]之一
Action Input: 工具的输入
Observation: <result>工具返回的结果</result>
Answer: 根据Observation总结本次工具调用返回的结果，如果结果中出现url，请使用如下格式展示出来：![图片](url)

"""

TOOL_TEMPLATE_ZH_PARALLEL = """
# 工具

## 你拥有如下工具：

{tool_descs}

## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令，可以根据需求调用零次或多次：

工具调用
Action: 工具1的名称，必须是[{tool_names}]之一
Action Input: 工具1的输入
Action: 工具2的名称，必须是[{tool_names}]之一
Action Input: 工具2的输入
...
Action: 工具N的名称，必须是[{tool_names}]之一
Action Input: 工具N的输入
Observation: <result>工具1返回的结果</result>
Observation: <result>工具2返回的结果</result>
...
Observation: <result>工具N返回的结果</result>

Answer: 根据Observation总结本次工具调用返回的结果，如果结果中出现url，请使用如下格式展示出来：![图片](url)

"""

TOOL_TEMPLATE_EN = """
# Tools

## You have the following tools:

{tool_descs}

## When you need to call a tool, please intersperse the following tool command in your reply. %s

Tool Invocation
Action: The name of the tool, must be one of [{tool_names}]
Action Input: Tool input
Observation: <result>Tool returns result</result>
Answer: Summarize the results of this tool call based on Observation. If the result contains url, %s

""" % ('You can call zero or more times according to your needs:',
       'please display it in the following format:![Image](URL)')

TOOL_TEMPLATE_EN_PARALLEL = """
# Tools

## You have the following tools:

{tool_descs}

## When you need to call a tool, please intersperse the following tool command in your reply. %s

Tool Invocation
Action: The name of the tool 1, must be one of [{tool_names}]
Action Input: Tool input ot tool 1
Action: The name of the tool 2, must be one of [{tool_names}]
Action Input: Tool input ot tool 2
...
Action: The name of the tool N, must be one of [{tool_names}]
Action Input: Tool input ot tool N
Observation: <result>Tool 1 returns result</result>
Observation: <result>Tool 1 returns result</result>
...
Observation: <result>Tool N returns result</result>
Answer: Summarize the results of this tool call based on Observation. If the result contains url, %s

""" % ('You can call zero or more times according to your needs:',
       'please display it in the following format:![Image](URL)')

TOOL_TEMPLATE = {
    'zh': TOOL_TEMPLATE_ZH,
    'en': TOOL_TEMPLATE_EN,
    'zh_parallel': TOOL_TEMPLATE_ZH_PARALLEL,
    'en_parallel': TOOL_TEMPLATE_EN_PARALLEL,
}

SPECIAL_PREFIX_TEMPLATE_TOOL = {
    'zh': '。你可以使用工具：[{tool_names}]',
    'en': '. you can use tools: [{tool_names}]',
}

SPECIAL_PREFIX_TEMPLATE_TOOL_FOR_CHAT = {
    'zh': '。你必须使用工具中的一个或多个：[{tool_names}]',
    'en': '. you must use one or more tools: [{tool_names}]',
}


def detect_multi_tool(message: Union[str, dict]) -> Tuple[bool, list, str]:
    """
    parse 'Action: xxx Action input: yyy\n\nAction: ppp Action input: qqq' into
    {'xxx': 'yyy', 'ppp': 'qqq'}
    Args:
        message:  str message only for now

    Returns:
        if contain tools, action and action input in a dict format, text string of the message
    """

    assert isinstance(message, str)
    text = message
    # find first Action
    match_result = re.findall(r'Action: (.+)\nAction Input: (.+)', text)

    tools = []
    for item in match_result:
        func_name, func_args = item
        tool_info = {'name': func_name, 'arguments': func_args}
        tools.append(tool_info)

    return (len(tools) > 0), tools, text


def convert_tools_to_prompt(tool_list: List) -> str:
    """
    convert action_dict to 'Action: xxx\nAction Input: yyyy\n\nAction: ppp\nAction Input: qqq'
    Args:
        tool_list: list of tools

    Returns:
        string of the tools
    """
    return '\n\n'.join([
        f'Action: {tool["name"]}\nAction Input: {tool["arguments"]}'
        for tool in tool_list
    ])
