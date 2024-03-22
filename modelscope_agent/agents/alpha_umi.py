import os
from typing import Dict, List, Optional, Tuple, Union

import json
from modelscope_agent import Agent
from modelscope_agent.llm import get_chat_model
from modelscope_agent.llm.base import BaseChatModel

PLANNER_TEMPLATE = """You have assess to the following apis:
{doc}
The conversation history is:
{history}
You are the assistant to plan what to do next and whether is caller's or conclusion's turn to answer.
Answer with a following format:
The thought of the next step, followed by Next: caller or conclusion or give up."""

CALLER_TEMPLATE = """You have assess to the following apis:
{doc}
The conversation history is:
{history}
The thought of this step is:
{thought}
Base on the thought make an api call in the following format:
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request."""

SUMMARIZER_TEMPLATE = """Make a conclusion based on the conversation history:
{history}"""

ACTION_TOKEN = 'Action:'
ARGS_TOKEN = 'Action Input:'
OBSERVATION_TOKEN = 'Observation:'
ANSWER_TOKEN = 'Answer:'


class AlphaUmi(Agent):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm_planner: Optional[Union[Dict, BaseChatModel]] = None,
                 llm_caller: Optional[Union[Dict, BaseChatModel]] = None,
                 llm_summarizer: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 **kwargs):
        """
        init tools/llm/instruction for one agent

        Args:
            function_list: A list of tools
                (1)When str: tool names
                (2)When Dict: tool cfg
            llm: The llm config of this agent
                (1) When Dict: set the config of llm as {'model': '', 'api_key': '', 'model_server': ''}
                (2) When BaseChatModel: llm is sent by another agent
            storage_path: If not specified otherwise, all data will be stored here in KV pairs by memory
            name: the name of agent
            description: the description of agent, which is used for multi_agent
            instruction: the system instruction of this agent
            kwargs: other potential parameters
        """
        """
        修改：
            1. 同时加载三个llm，要传3个config
        """
        try:
            import vllm
        except ImportError:
            raise ImportError(
                'The vllm package is not installed.'
                'Please make sure GPU env is ready.'
                'Refer to https://docs.vllm.ai/en/latest/getting_started/installation.html'
            )
        if isinstance(llm_planner, Dict):
            self.llm_config_planner = llm_planner
            self.llm_planner = get_chat_model(**self.llm_config_planner)
        else:
            self.llm_planner = llm_planner

        if isinstance(llm_caller, Dict):
            self.llm_config_caller = llm_caller
            self.llm_caller = get_chat_model(**self.llm_config_caller)
        else:
            self.llm_caller = llm_caller

        if isinstance(llm_summarizer, Dict):
            self.llm_config_summarizer = llm_summarizer
            self.llm_summarizer = get_chat_model(**self.llm_config_summarizer)
        else:
            self.llm_summarizer = llm_summarizer
        self.stream = True

        self.function_list = []
        self.function_map = {}
        if function_list:
            for function in function_list:
                self._register_tool(function)

        self.storage_path = storage_path
        self.mem = None
        self.name = name
        self.description = description
        self.instruction = instruction
        self.uuid_str = kwargs.get('uuid_str', None)

    def _run(self,
             user_request,
             history: Optional[List[Dict]] = None,
             ref_doc: str = None,
             lang: str = 'zh',
             **kwargs):
        """
        修改：
        1. 取消了message，改为用prompt. 多步生成的对话历史用history控制
        2. 底层在调用LLM的时候直接传prompt，但是prompt的前面不要加role:user，
            相当于在底层llm.chat的实现时直接把llm.chat传入的prompt参数作为模型输入就行，不要加任何额外信息
        3. 不需要验证llm.support_function_calling，默认llm_caller是一定会生成Action: Action Input:的
        """

        self.tool_descs = '\n'.join(tool.function_plain_text
                                    for tool in self.function_map.values())
        self.tool_names = ', '.join(tool.name
                                    for tool in self.function_map.values())

        self.planner_prompt = PLANNER_TEMPLATE.replace('{doc}',
                                                       self.tool_descs)
        self.caller_prompt = CALLER_TEMPLATE.replace(
            '{doc}', self.tool_descs).replace('{tool_names}', self.tool_names)
        self.summarizer_prompt = SUMMARIZER_TEMPLATE
        # Concat the system as one round of dialogue

        if history:
            assert history[-1][
                'role'] != 'user', 'The history should not include the latest user query.'
            if history[0]['role'] == 'system':
                history = history[1:]
        else:
            history = list()
        history.append({'role': 'user', 'content': user_request})

        # concat the new messages
        max_turn = 10
        while True and max_turn > 0:
            dispatch_history = self._concat_history(history)
            max_turn -= 1
            planner_output = self.llm_planner.chat(
                prompt=self.planner_prompt.replace(
                    '{history}', dispatch_history) + ' assistant: ',
                max_tokens=2000,
                stream=False,
                **kwargs)

            decision, planner_result = self._parse_planner_output(
                planner_output)
            history.append({'role': 'assistant', 'content': planner_output})
            yield planner_output

            if decision == 'give_up':
                break

            elif decision == 'caller':
                dispatch_history = self._concat_history(history)

                caller_output = self.llm_caller.chat(
                    prompt=self.caller_prompt.replace(
                        '{history}', dispatch_history).replace(
                            '{thought}', history[-1]['content']) + ' caller: ',
                    stream=False,
                    max_tokens=2000,
                    **kwargs)

                use_tool, action, action_input, caller_output = self._detect_tool(
                    caller_output)

                history.append({'role': 'caller', 'content': caller_output})
                yield caller_output

                if use_tool:
                    yield f'Action: {action}\nAction Input: {action_input}'
                    observation = self._call_tool(action, action_input)
                    yield f'Observation: {observation}'
                    if isinstance(observation, dict) or isinstance(
                            observation, list):
                        observation_str = json.dumps(observation)
                    elif isinstance(observation, str):
                        observation_str = observation
                    else:
                        observation_str = str(observation)
                    history.append({
                        'role': 'observation',
                        'content': observation_str
                    })
            else:
                dispatch_history = self._concat_history(history)
                summarizer_output = self.llm_summarizer.chat(
                    prompt=self.summarizer_prompt.replace(
                        '{history}', dispatch_history) + ' conclusion: ',
                    stream=False,
                    max_tokens=2000,
                    **kwargs)
                yield summarizer_output

                history.append({
                    'role': 'conclusion',
                    'content': summarizer_output
                })
                break

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

        if func_name is not None:
            find_tool = False
            for tool in self.function_map.values():
                if tool.name.endswith(func_name):
                    func_name = tool.name
                    find_tool = True
                    break

        return (func_name is not None
                and find_tool), func_name, func_args, text

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

    def _concat_history(self, history):
        res = ''
        for utter in history:
            if not isinstance(utter, dict) or not utter.get('role', None):
                continue
            if utter['role'] == 'assistant':
                res += ('assistant: ' + utter['content'] + '</s>')
            elif utter['role'] == 'user':
                res += ('user: ' + utter['content'] + '</s>')
            elif utter['role'] == 'observation':
                res += ('observation: ' + utter['content'])
            elif utter['role'] == 'caller':
                res += ('caller: ' + utter['content'] + '</s>')
            elif utter['role'] == 'conclusion':
                res += ('conclusion: ' + utter['content'] + '</s>')
        return res

    def _parse_planner_output(self, planner_output):
        assert isinstance(planner_output, str)
        if 'Next: give up.' in planner_output:
            action_end_idx = planner_output.index('Next: give up.')
            planner_output = planner_output[:action_end_idx
                                            + len('Next: give up.')]
            return 'give_up', planner_output
        elif 'Next: caller.' in planner_output:
            action_end_idx = planner_output.index('Next: caller.')
            planner_output = planner_output[:action_end_idx
                                            + len('Next: caller.')]
            return 'caller', planner_output
        else:
            if 'Next: conclusion.' in planner_output:
                action_end_idx = planner_output.index('Next: conclusion.')
                planner_output = planner_output[:action_end_idx
                                                + len('Next: conclusion.')]
            else:
                planner_output = planner_output + 'Next: conclusion.'
            return 'summarizer', planner_output

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
