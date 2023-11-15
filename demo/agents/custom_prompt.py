DEFAULT_SYSTEM_TEMPLATE = """
# 工具

## 你拥有如下工具：

<tool_list>

## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令：

```工具调用
Action: 工具的名字
Action Input: 工具的输入，需格式化为一个JSON
Observation: 工具返回的结果
```

# 指令
"""

DEFAULT_USER_TEMPLATE = """\n<user_input>\n"""

DEFAULT_EXEC_TEMPLATE = """Observation: <exec_result>```"""
