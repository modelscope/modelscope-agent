import os
import re
from typing import Dict, List, Optional, Tuple, Union

from modelscope_agent import Agent
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.utils.tokenization_utils import count_tokens
from modelscope_agent.utils.utils import check_and_limit_input_length

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
{role_prompt}
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

PROMPT_TEMPLATE_EN = """{role_prompt}"""

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
    'zh': '[上传文件 "{file_names}"]',
    'en': '[Upload file "{file_names}"]',
}

DEFAULT_EXEC_TEMPLATE = """\nObservation: <result>{exec_result}</result>\nAnswer:"""

ACTION_TOKEN = 'Action:'
ARGS_TOKEN = 'Action Input:'
OBSERVATION_TOKEN = 'Observation:'
ANSWER_TOKEN = 'Answer:'


class MultiRolePlay(RolePlay):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

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
            knowledge_limit = kwargs.get('knowledge_limit',
                                         os.getenv('KNOWLEDGE_LIMIT', 4000))
            ref_doc = check_and_limit_input_length(ref_doc, knowledge_limit)
            self.system_prompt += KNOWLEDGE_TEMPLATE[lang].format(
                ref_doc=ref_doc)
            self.query_prefix_dict[
                'knowledge'] = SPECIAL_PREFIX_TEMPLATE_KNOWLEDGE[lang]

        # concat tools information
        if self.function_map and not self.llm.support_function_calling():
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

        self.query_prefix = ''
        self.query_prefix += self.query_prefix_dict['role']
        self.query_prefix += self.query_prefix_dict['tool']
        self.query_prefix += self.query_prefix_dict['knowledge']
        if self.query_prefix:
            self.query_prefix = '(' + self.query_prefix + ')'

        if len(append_files) > 0:
            file_names = ','.join(
                [os.path.basename(path) for path in append_files])
            self.query_prefix += SPECIAL_PREFIX_TEMPLATE_FILE[lang].format(
                file_names=file_names)

        # Concat the system as one round of dialogue
        messages = [{'role': 'system', 'content': self.system_prompt}]
        print('history: ', history)
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
        print('user_request', user_request)
        planning_prompt = ''
        if self.llm.support_raw_prompt():
            planning_prompt = self.llm.build_multi_role_raw_prompt(messages)

        max_turn = 10
        while True and max_turn > 0:
            max_turn -= 1
            if self.llm.support_function_calling():
                output = self.llm.chat_with_functions(
                    messages=messages,
                    stream=True,
                    functions=[
                        func.function for func in self.function_map.values()
                    ],
                    **kwargs)
            else:
                output = self.llm.chat(
                    prompt=planning_prompt,
                    stream=True,
                    stop=['Observation:', 'Observation:\n'],
                    messages=messages,
                    **kwargs)

            llm_result = ''
            for s in output:
                if isinstance(s, dict):
                    llm_result = s
                    break
                else:
                    llm_result += s
                yield s

            if isinstance(llm_result, str):
                use_tool, action, action_input, output = self._detect_tool(
                    llm_result)
            elif isinstance(llm_result, dict):
                use_tool, action, action_input, output = super()._detect_tool(
                    llm_result)
            else:
                assert 'llm_result must be an instance of dict or str'

            # yield output
            if use_tool:
                if self.llm.support_function_calling():
                    yield f'Action: {action}\nAction Input: {action_input}'
                observation = self._call_tool(action, action_input)
                format_observation = DEFAULT_EXEC_TEMPLATE.format(
                    exec_result=observation)
                yield format_observation
                format_observation = self._limit_observation_length(
                    format_observation)
                if self.llm.support_function_calling():
                    messages.append({'role': 'tool', 'content': observation})
                else:
                    messages[-1]['content'] += output + format_observation
                    planning_prompt += output + format_observation

            else:
                messages[-1]['content'] += output
                planning_prompt += output
                break

            # limit the length of the planning prompt if exceed the length by calling the build_raw_prompt
            if not self.llm.check_max_length(planning_prompt):
                if self.llm.support_raw_prompt():
                    planning_prompt = self.llm.build_multi_role_raw_prompt(
                        messages)
