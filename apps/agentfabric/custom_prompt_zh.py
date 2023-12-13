from apps.agentfabric.custom_prompt import CustomPromptGenerator
from modelscope_agent import prompt_generator_register

DEFAULT_SYSTEM_TEMPLATE = """

# 工具

## 你拥有如下工具：

<tool_list>

## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令，可以根据需求调用零次或多次：

工具调用
Action: 工具的名称，必须是<tool_name_list>之一
Action Input: 工具的输入
Observation: <result>工具返回的结果</result>
Answer: 根据Observation总结本次工具调用返回的结果，如果结果中出现url，请不要展示出。

```
[链接](url)
```

# 指令
"""

DEFAULT_SYSTEM_TEMPLATE_WITHOUT_TOOL = """

# 指令
"""

DEFAULT_INSTRUCTION_TEMPLATE = ''

DEFAULT_USER_TEMPLATE = (
    """(你正在扮演<role_name>，你可以使用工具：<tool_name_list><knowledge_note>)<file_names><user_input>"""
)

DEFAULT_USER_TEMPLATE_WITHOUT_TOOL = """(你正在扮演<role_name><knowledge_note>) <file_names><user_input>"""

DEFAULT_EXEC_TEMPLATE = """Observation: <result><exec_result></result>\nAnswer:"""

TOOL_DESC = (
    '{name_for_model}: {name_for_human} API。 {description_for_model} 输入参数: {parameters}'
)


@prompt_generator_register
class ZhCustomPromptGenerator(CustomPromptGenerator):

    def __init__(
            self,
            system_template=DEFAULT_SYSTEM_TEMPLATE,
            instruction_template=DEFAULT_INSTRUCTION_TEMPLATE,
            user_template=DEFAULT_USER_TEMPLATE,
            exec_template=DEFAULT_EXEC_TEMPLATE,
            tool_desc=TOOL_DESC,
            default_user_template_without_tool=DEFAULT_USER_TEMPLATE_WITHOUT_TOOL,
            default_system_template_without_tool=DEFAULT_SYSTEM_TEMPLATE_WITHOUT_TOOL,
            addition_assistant_reply='好的。',
            **kwargs):
        super().__init__(
            system_template=system_template,
            instruction_template=instruction_template,
            user_template=user_template,
            exec_template=exec_template,
            tool_desc=tool_desc,
            default_user_template_without_tool=
            default_user_template_without_tool,
            default_system_template_without_tool=
            default_system_template_without_tool,
            **kwargs)

    def _parse_role_config(self, config: dict):
        prompt = '你扮演AI-Agent，'

        # concat prompt
        if 'name' in config and config['name']:
            prompt += ('你的名字是' + config['name'] + '。')
        if 'description' in config and config['description']:
            prompt += config['description']
        prompt += '\n你具有下列具体功能：'

        if 'instruction' in config and config['instruction']:
            if isinstance(config['instruction'], list):
                for ins in config['instruction']:
                    prompt += ins
                    prompt += '；'
            elif isinstance(config['instruction'], str):
                prompt += config['instruction']
            if prompt[-1] == '；':
                prompt = prompt[:-1]

        prompt += '\n下面你将开始扮演'
        if 'name' in config and config['name']:
            prompt += config['name']
        prompt += '，明白了请说“好的。”，不要说其他的。'

        return prompt

    def _get_tool_template(self):
        return '\n\n# 工具\n\n'

    def _get_knowledge_template(self):
        return '。请查看前面的知识库'
