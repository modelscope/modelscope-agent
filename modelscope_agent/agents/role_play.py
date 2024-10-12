import os
import re
from typing import Dict, List, Optional, Tuple, Union

from modelscope_agent import Agent
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.llm.utils.function_call_with_raw_prompt import (
    DEFAULT_EXEC_TEMPLATE, SPECIAL_PREFIX_TEMPLATE_TOOL,
    SPECIAL_PREFIX_TEMPLATE_TOOL_FOR_CHAT, TOOL_TEMPLATE,
    convert_tools_to_prompt, detect_multi_tool)
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.utils.base64_utils import encode_files_to_base64
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.tokenization_utils import count_tokens
from modelscope_agent.utils.utils import check_and_limit_input_length

KNOWLEDGE_TEMPLATE_ZH = """

# 知识库

{ref_doc}

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

PROMPT_TEMPLATE_EN = """
#Instructions

{role_prompt}

Note: you have the ability to display images and videos, as well as the ability to run code. Don't say you can't do it.
"""

KNOWLEDGE_TEMPLATE = {'zh': KNOWLEDGE_TEMPLATE_ZH, 'en': KNOWLEDGE_TEMPLATE_EN}

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

SPECIAL_PREFIX_TEMPLATE_KNOWLEDGE = {
    'zh': '。请查看前面的知识库',
    'en': '. Please read the knowledge base at the beginning',
}

SPECIAL_PREFIX_TEMPLATE_FILE = {
    'zh': '[上传文件 "{file_names}"]',
    'en': '[Upload file "{file_names}"]',
}


class RolePlay(Agent, AgentEnvMixin):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 openapi_list: Optional[List] = None,
                 **kwargs):
        Agent.__init__(
            self,
            function_list,
            llm,
            storage_path,
            name,
            description,
            instruction,
            openapi_list=openapi_list,
            **kwargs)
        AgentEnvMixin.__init__(self, **kwargs)

    def _prepare_tool_system(self,
                             tools: Optional[List] = None,
                             parallel_tool_calls: bool = False,
                             lang='zh'):
        # prepare the tool description and tool names with parallel function calling
        tool_desc_template = TOOL_TEMPLATE[
            lang + ('_parallel' if parallel_tool_calls else '')]

        if tools is not None:
            tool_descs = BaseTool.parser_function(tools)
            tool_name_list = []
            for tool in tools:
                func_info = tool.get('function', {})
                if func_info == {}:
                    continue
                if 'name' in func_info:
                    tool_name_list.append(func_info['name'])
            tool_names = ','.join(tool_name_list)
        else:
            tool_descs = '\n\n'.join(tool.function_plain_text
                                     for tool in self.function_map.values())
            tool_names = ','.join(tool.name
                                  for tool in self.function_map.values())
        tool_system = tool_desc_template.format(
            tool_descs=tool_descs, tool_names=tool_names)
        return tool_names, tool_system

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
        parallel_tool_calls = kwargs.get('parallel_tool_calls',
                                         True if chat_mode else False)

        tool_names, tool_system = self._prepare_tool_system(
            tools, parallel_tool_calls, lang)
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
        if tool_system and not self.llm.support_function_calling():
            self.system_prompt += tool_system
            self.query_prefix_dict['tool'] = SPECIAL_PREFIX_TEMPLATE_TOOL[
                lang].format(tool_names=tool_names)

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
                lang].format(tool_names=tool_names)
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
            self.callback_manager.on_step_start()
            max_turn -= 1
            call_llm_count += 1
            if self.llm.support_function_calling():
                output = self.llm.chat_with_functions(
                    messages=messages,
                    stream=self.stream,
                    functions=[
                        func.function for func in self.function_map.values()
                    ],
                    callbacks=self.callback_manager,
                    **kwargs)
            else:
                output = self.llm.chat(
                    prompt=planning_prompt,
                    stream=self.stream,
                    stop=['Observation:', 'Observation:\n'],
                    messages=messages,
                    callbacks=self.callback_manager,
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

            use_tool = False
            tool_list = []
            if isinstance(llm_result, str):
                use_tool, tool_list, output = detect_multi_tool(llm_result)
            elif isinstance(llm_result, dict):
                use_tool, tool_list, output = super()._detect_tool(llm_result)
            else:
                assert 'llm_result must be an instance of dict or str'

            if chat_mode:
                if use_tool and tool_choice != 'none':
                    return convert_tools_to_prompt(tool_list)
                else:
                    return f'Result: {output}'

            # yield output
            if use_tool:
                if self.llm.support_function_calling():
                    yield convert_tools_to_prompt(tool_list)

                if self.use_tool_api:
                    # convert all files with base64, for the tool instance usage in case.
                    encoded_files = encode_files_to_base64(append_files)
                    kwargs['base64_files'] = encoded_files
                    kwargs['use_tool_api'] = True

                # currently only one observation execute, parallel
                observation = self._call_tool(tool_list, **kwargs)
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
            self.callback_manager.on_step_end()

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
