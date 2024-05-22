import os
import re
from typing import Dict, List, Optional, Tuple, Union

from modelscope_agent import Agent
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.utils.logger import agent_logger as logger
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

SPECIAL_PREFIX_TEMPLATE_TOOL_FOR_CHAT = {
    'zh': '。你必须使用工具中的一个或多个：[{tool_names}]',
    'en': '. you must use one or more tools: [{tool_names}]',
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


class RolePlay(Agent, AgentEnvMixin):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 **kwargs):
        Agent.__init__(self, function_list, llm, storage_path, name,
                       description, instruction, **kwargs)
        AgentEnvMixin.__init__(self, **kwargs)

    def _run(self,
             user_request,
             history: Optional[List[Dict]] = None,
             ref_doc: str = None,
             image_url: Optional[List[Union[str, Dict]]] = None,
             lang: str = 'zh',
             **kwargs):

        chat_mode = kwargs.pop('chat_mode', False)
        tools = kwargs.get('tools', None)
        tool_choice = kwargs.get('tool_choice', 'auto')

        if tools is not None:
            self.tool_descs = BaseTool.parser_function(tools)
            tool_name_list = []
            for tool in tools:
                func_info = tool.get('function', {})
                if func_info == {}:
                    continue
                if 'name' in func_info:
                    tool_name_list.append(func_info['name'])
            self.tool_names = ','.join(tool_name_list)
        else:
            self.tool_descs = '\n\n'.join(
                tool.function_plain_text
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
        if self.tool_descs and not self.llm.support_function_calling():
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

        if history:
            assert history[-1][
                'role'] != 'user', 'The history should not include the latest user query.'
            if history[0]['role'] == 'system':
                history = history[1:]
            messages.extend(history)

        # concat the new messages
        if chat_mode and tool_choice == 'required':
            required_prefix = SPECIAL_PREFIX_TEMPLATE_TOOL_FOR_CHAT[
                lang].format(tool_names=self.tool_names)
            messages.append({
                'role': 'user',
                'content': required_prefix + user_request
            })
        else:
            messages.append({
                'role': 'user',
                'content': self.query_prefix + user_request
            })

        if image_url:
            self._parse_image_url(image_url, messages)

        planning_prompt = ''
        if self.llm.support_raw_prompt() and hasattr(self.llm,
                                                     'build_raw_prompt'):
            planning_prompt = self.llm.build_raw_prompt(messages)

        max_turn = 10
        call_llm_count = 0
        while True and max_turn > 0:
            # print('=====one input planning_prompt======')
            # print(planning_prompt)
            # print('=============Answer=================')
            max_turn -= 1
            call_llm_count += 1
            if self.llm.support_function_calling():
                output = self.llm.chat_with_functions(
                    messages=messages,
                    stream=self.stream,
                    functions=[
                        func.function for func in self.function_map.values()
                    ],
                    **kwargs)
            else:
                output = self.llm.chat(
                    prompt=planning_prompt,
                    stream=self.stream,
                    stop=['Observation:', 'Observation:\n'],
                    messages=messages,
                    **kwargs)

            llm_result = ''
            logger.info(f'call llm {call_llm_count} times output: {output}')
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

            if chat_mode:
                if use_tool and tool_choice != 'none':
                    return f'Action: {action}\nAction Input: {action_input}\nResult: {output}'
                else:
                    return f'Result: {output}'

            # yield output
            if use_tool:
                if self.llm.support_function_calling():
                    yield f'Action: {action}\nAction Input: {action_input}'
                observation = self._call_tool(action, action_input, **kwargs)
                format_observation = DEFAULT_EXEC_TEMPLATE.format(
                    exec_result=observation)
                yield format_observation

                # for next turn
                format_observation = self._limit_observation_length(
                    observation)
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
                    planning_prompt = self.llm.build_raw_prompt(messages)

    def _limit_observation_length(self, observation):
        """
        limit the observation result length if exceeds half of the max length
        Args:
            observation: the output from the tool

        Returns:

        """
        reasonable_length = self.llm.get_max_length() / 2 - count_tokens(
            DEFAULT_EXEC_TEMPLATE.format(exec_result=' '))
        limited_observation = str(observation)[:int(reasonable_length)]
        return DEFAULT_EXEC_TEMPLATE.format(exec_result=limited_observation)

    def _detect_tool(self, message: Union[str,
                                          dict]) -> Tuple[bool, str, str, str]:
        assert isinstance(message, str)
        text = message
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
        """
        Parsing role config dict to str.

        Args:
            config: One example of config is
                {
                    "name": "多啦A梦",
                    "description": "能够像多啦A梦一样，拥有各种神奇的技能和能力，可以帮我解决生活中的各种问题。",
                    "instruction": "可以查找信息、提供建议、提醒日程；爱讲笑话，每次说话的结尾都会加上一句幽默的总结；最喜欢的人是大熊"
                }
        Returns:
            Processed string for this config
        """
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
