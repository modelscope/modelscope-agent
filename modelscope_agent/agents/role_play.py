import os
from typing import Dict, List, Optional

from modelscope_agent import Agent

KNOWLEDGE_TEMPLATE_ZH = """

# 知识库

{ref_doc}

"""

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

PROMPT_TEMPLATE_ZH = """
# 指令

{role_prompt}

请注意：你具有图像和视频的展示能力，也具有运行代码的能力，不要在回复中说你做不到。
"""

KNOWLEDGE_TEMPLATE_EN = """

# Knowledge Base

{ref_doc}

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

PROMPT_TEMPLATE_EN = """
#Instructions

{role_prompt}

Note: you have the ability to display images and videos, as well as the ability to run code. Don't say you can't do it.
"""

KNOWLEDGE_TEMPLATE = {'zh': KNOWLEDGE_TEMPLATE_ZH, 'en': KNOWLEDGE_TEMPLATE_EN}

TOOL_TEMPLATE = {
    'zh': TOOL_TEMPLATE_ZH,
    'en': TOOL_TEMPLATE_EN,
}

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}

PREFIX_PROMPT_TEMPLATE = {
    'zh': '，明白了请说“好的。”，不要说其他的。',
    'en': ', say "OK." if you understand, do not say anything else.'
}

SYSTEM_ANSWER_TEMPLATE = {
    'zh': '好的。',
    'en': 'OK.',
}

SPECIAL_PREFIX_TEMPLATE_ROLE = {
    'zh': '你正在扮演{role_name}',
    'en': 'You are playing as {role_name}',
}

SPECIAL_PREFIX_TEMPLATE_TOOL = {
    'zh': '。你可以使用工具：[{tool_names}]',
    'en': '. you can use tools: [{tool_names}]',
}

SPECIAL_PREFIX_TEMPLATE_KNOWLEDGE = {
    'zh': '。请查看前面的知识库',
    'en': '. Please read the knowledge base at the beginning',
}

SPECIAL_PREFIX_TEMPLATE_FILE = {
    'zh': '[上传文件{file_names}]',
    'en': '[Upload file {file_names}]',
}

DEFAULT_EXEC_TEMPLATE = """\nObservation: <result>{exec_result}</result>\nAnswer:"""

ACTION_TOKEN = 'Action:'
ARGS_TOKEN = 'Action Input:'
OBSERVATION_TOKEN = 'Observation:'
ANSWER_TOKEN = 'Answer:'


class RolePlay(Agent):

    def _run(self,
             user_request,
             history: Optional[List[Dict]] = None,
             ref_doc: str = None,
             lang: str = 'zh',
             **kwargs):

        self.tool_descs = '\n\n'.join(tool.function_plain_text
                                      for tool in self.function_map.values())
        self.tool_names = ','.join(tool.name
                                   for tool in self.function_map.values())

        self.system_prompt = ''
        self.query_prefix = ''
        self.query_prefix_dict = {'role': '', 'tool': '', 'knowledge': ''}
        append_files = kwargs.get('append_files', [])

        # code interpreter might be not work with ref_doc at the same time, comment on 2024-01-10
        use_ref_doc = True
        if len(append_files) > 0 and 'code_interpreter' in self.function_map:
            use_ref_doc = False

        # concat knowledge
        if ref_doc and use_ref_doc:
            self.system_prompt += KNOWLEDGE_TEMPLATE[lang].format(
                ref_doc=ref_doc)
            self.query_prefix_dict[
                'knowledge'] = SPECIAL_PREFIX_TEMPLATE_KNOWLEDGE[lang]

        # concat tools information
        if self.function_map:
            self.system_prompt += TOOL_TEMPLATE[lang].format(
                tool_descs=self.tool_descs, tool_names=self.tool_names)
            self.query_prefix_dict['tool'] = SPECIAL_PREFIX_TEMPLATE_TOOL[
                lang].format(tool_names=self.tool_names)

        # concat instruction
        if isinstance(self.instruction, dict):
            self.role_name = self.instruction['name']
            self.query_prefix_dict['role'] = SPECIAL_PREFIX_TEMPLATE_ROLE[
                lang].format(role_name=self.role_name)
            self.system_prompt += PROMPT_TEMPLATE[lang].format(
                role_prompt=self._parse_role_config(self.instruction, lang))
        else:
            # string can not parser role name
            self.role_name = ''
            self.system_prompt += PROMPT_TEMPLATE[lang].format(
                role_prompt=self.instruction)

        self.query_prefix = '('
        self.query_prefix += self.query_prefix_dict['role']
        self.query_prefix += self.query_prefix_dict['tool']
        self.query_prefix += self.query_prefix_dict['knowledge']
        self.query_prefix += ')'

        if len(append_files) > 0:
            file_names = ','.join(
                [os.path.basename(path) for path in append_files])
            self.query_prefix += SPECIAL_PREFIX_TEMPLATE_FILE[lang].format(
                file_names=file_names)

        # Concat the system as one round of dialogue
        messages = [{'role': 'system', 'content': self.system_prompt}]

        if history:
            assert history[-1][
                'role'] != 'user', 'The history should not include the latest user query.'
            if history[0]['role'] == 'system':
                history = history[1:]
            messages.extend(history)

        # concat the new messages
        messages.append({
            'role': 'user',
            'content': self.query_prefix + user_request
        })
        messages.append({'role': 'assistant', 'content': ''})

        planning_prompt = self.llm.build_raw_prompt(messages)

        max_turn = 10
        while True and max_turn > 0:
            # print('=====one input planning_prompt======')
            # print(planning_prompt)
            # print('=============Answer=================')
            max_turn -= 1
            output = self.llm.chat(
                prompt=planning_prompt,
                stream=True,
                stop=['Observation:', 'Observation:\n'],
                messages=messages,
                **kwargs)

            llm_result = ''
            for s in output:
                llm_result += s
                yield s

            use_tool, action, action_input, output = self._detect_tool(
                llm_result)

            # yield output
            print(output)
            if use_tool:
                observation = self._call_tool(action, action_input)
                observation = DEFAULT_EXEC_TEMPLATE.format(
                    exec_result=observation)
                yield observation
                print(observation)
                planning_prompt += output + observation
            else:
                planning_prompt += output
                break

    def _detect_tool_by_special_token(self, text: str):
        func_name, func_args = None, None
        i = text.rfind(ACTION_TOKEN)
        j = text.rfind(ARGS_TOKEN)
        k = text.rfind(OBSERVATION_TOKEN)
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ommited by the LLM,
                # because the output text may have discarded the stop word.
                text = text.rstrip() + OBSERVATION_TOKEN  # Add it back.
            k = text.rfind(OBSERVATION_TOKEN)
            func_name = text[i + len(ACTION_TOKEN):j].strip()
            func_args = text[j + len(ARGS_TOKEN):k].strip()
            text = text[:k]  # Discard '\nObservation:'.

        return (func_name is not None), func_name, func_args, text

    def _parse_role_config(self, config: dict, lang: str = 'zh') -> str:
        if lang == 'en':
            return self._parse_role_config_en(config)
        else:
            return self._parse_role_config_zh(config)

    def _parse_role_config_en(self, config: dict) -> str:
        prompt = 'You are playing as an AI-Agent, '

        # concat agents
        if 'name' in config and config['name']:
            prompt += ('Your name is ' + config['name'] + '.')
        if 'description' in config and config['description']:
            prompt += config['description']
        prompt += '\nYou have the following specific functions:'

        if 'instruction' in config and config['instruction']:
            if isinstance(config['instruction'], list):
                for ins in config['instruction']:
                    prompt += ins
                    prompt += '；'
            elif isinstance(config['instruction'], str):
                prompt += config['instruction']
            if prompt[-1] == '；':
                prompt = prompt[:-1]

        prompt += '\nNow you will start playing as'
        if 'name' in config and config['name']:
            prompt += config['name']

        return prompt

    def _parse_role_config_zh(self, config: dict) -> str:
        prompt = '你扮演AI-Agent，'

        # concat agents
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

        return prompt
