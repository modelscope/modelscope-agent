import copy
import os
import re

import json
from config_utils import DEFAULT_BUILDER_CONFIG_FILE
from modelscope_agent.prompt.prompt import PromptGenerator, build_raw_prompt

from modelscope.utils.config import Config

DEFAULT_SYSTEM_TEMPLATE = """# 工具

## 你拥有如下工具：

<tool_list>

## 当你需要调用工具时，请在你的回复中穿插如下的工具调用命令：

工具调用
Action: 工具的名字
Action Input: 工具的输入，需格式化为一个JSON
Observation: 工具返回的结果

# 指令
"""
DEFAULT_INSTRUCTION_TEMPLATE = ''

DEFAULT_USER_TEMPLATE = """(你正在扮演<role_name>，你可以使用工具：<tool_name_list>) <user_input>"""

DEFAULT_EXEC_TEMPLATE = """Observation: <exec_result>"""

TOOL_DESC = (
    '{name_for_model}: {name_for_human} API。 {description_for_model} 输入参数: {parameters}'
)


class CustomPromptGenerator(PromptGenerator):

    def __init__(self,
                 system_template=DEFAULT_SYSTEM_TEMPLATE,
                 instruction_template=DEFAULT_INSTRUCTION_TEMPLATE,
                 user_template=DEFAULT_USER_TEMPLATE,
                 exec_template=DEFAULT_EXEC_TEMPLATE,
                 assistant_template='',
                 sep='\n',
                 prompt_max_length=1000,
                 **kwargs):
        super().__init__(system_template, instruction_template, user_template,
                         exec_template, assistant_template, sep,
                         prompt_max_length)
        # hack here for special prompt, such as add an addition round before user input
        self.add_addition_round = kwargs.get('add_addition_round', False)
        self.addition_assistant_reply = kwargs.get('addition_assistant_reply',
                                                   '')
        builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                     DEFAULT_BUILDER_CONFIG_FILE)
        builder_cfg = Config.from_file(builder_cfg_file)
        self.builder_cfg = builder_cfg

    def init_prompt(self, task, tool_list, knowledge_list, llm_model):
        if len(self.history) == 0:

            self.history.append({
                'role': 'system',
                'content': 'You are a helpful assistant.'
            })

            self.prompt_preprocessor = build_raw_prompt(llm_model)

            prompt = f'{self.system_template}\n{self.instruction_template}'

            if len(knowledge_list) > 0:

                knowledge_str = self.get_knowledge_str(knowledge_list)
                # knowledge
                prompt = knowledge_str + prompt

            # get tool description str
            tool_str = self.get_tool_str(tool_list)
            prompt = prompt.replace('<tool_list>', tool_str)

            # user input
            user_input = self.user_template.replace('<user_input>', task)
            user_input = user_input.replace('<role_name>',
                                            self.builder_cfg.name)
            user_input = user_input.replace(
                '<tool_name_list>',
                ','.join([tool.name for tool in tool_list]))

            self.system_prompt = copy.deepcopy(prompt)

            # build history
            if self.add_addition_round:
                self.history.append({
                    'role': 'user',
                    'content': self.system_prompt
                })
                self.history.append({
                    'role': 'assistant',
                    'content': self.addition_assistant_reply
                })
                self.history.append({'role': 'user', 'content': user_input})
                self.history.append({
                    'role': 'assistant',
                    'content': self.assistant_template
                })
            else:
                self.history.append({
                    'role': 'user',
                    'content': self.system_prompt + user_input
                })
                self.history.append({
                    'role': 'assistant',
                    'content': self.assistant_template
                })

            self.function_calls = self.get_function_list(tool_list)
        else:
            self.history.append({
                'role':
                'user',
                'content':
                self.user_template.replace('<user_input>', task)
            })
            self.history.append({
                'role': 'assistant',
                'content': self.assistant_template
            })
        return self.system_prompt

    def get_tool_str(self, tool_list):
        tool_texts = []
        for tool in tool_list:
            tool_texts.append(
                TOOL_DESC.format(
                    name_for_model=tool.name,
                    name_for_human=tool.name,
                    description_for_model=tool.description,
                    parameters=json.dumps(tool.parameters,
                                          ensure_ascii=False)))
            # + ' ' + FORMAT_DESC['json'])
        tool_str = '\n\n'.join(tool_texts)
        return tool_str

    def _generate(self, llm_result, exec_result: str):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """
        if len(llm_result) != 0:
            self.history[-1]['content'] += f'{llm_result}'
        if len(exec_result) != 0:
            # handle image markdown wrapper
            image_markdwon_re = re.compile(
                pattern=r'!\[IMAGEGEN\]\(([\s\S]+)\)')
            match = image_markdwon_re.search(exec_result)
            if match is not None:
                exec_result = match.group(1).rstrip()
            exec_result = self.exec_template.replace('<exec_result>',
                                                     str(exec_result))
            self.history[-1]['content'] += exec_result

        # generate plate prompt here
        self.prompt = self.prompt_preprocessor(self.history)
        return self.prompt


def parse_role_config(config: dict):
    prompt = '你扮演AI助手，'

    # replace special words
    for key in config:
        if isinstance(config[key], str):
            config[key] = config[key].replace('AI-Agent', 'AI助手')
        elif isinstance(config[key], list):
            for i in range(len(config[key])):
                config[key][i] = config[key][i].replace('AI-Agent', 'AI助手')
        else:
            pass

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
