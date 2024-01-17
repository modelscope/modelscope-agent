from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.llm import get_chat_model
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.tools import TOOL_REGISTRY
from modelscope_agent.utils.utils import has_chinese_chars
from modelscope_agent.utils.logger import agent_logger as logger


class Agent(ABC):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 **kwargs):
        """
        init tools/llm for one agent

        :param function_list: Optional[List[Union[str, Dict]]] :
            (1)When str: tool names
            (2)When Dict: tool cfg
        :param llm: Optional[Union[Dict, BaseChatModel]]:
            (1) When Dict: set the config of llm as {'model': '', 'api_key': '', 'model_server': ''}
            (2) When BaseChatModel: llm is sent by another agent
        :param storage_path: If not specified otherwise, all data will be stored here in KV pairs by memory
        :param name: the name of agent
        :param description: the description of agent, which is used for multi_agent
        :param instruction: the system instruction of this agent
        :param kwargs: other potential parameters
        """
        if isinstance(llm, Dict):
            self.llm_config = llm
            self.llm = get_chat_model(**self.llm_config)
        else:
            self.llm = llm
        self.stream = True

        self.function_list = []
        self.function_map = {}
        if function_list:
            not_implemented = []
            for function in function_list:
                try:
                    self._register_tool(function)
                except Exception:
                    not_implemented.append(function)
            if not_implemented:
                logger.query_warning(
                    uuid=kwargs.get('uuid_str', 'local_user'),
                    details=str(not_implemented),
                    message=f'Not implemented tool(s): {not_implemented}.')

        self.storage_path = storage_path
        self.mem = None
        self.name = name
        self.description = description
        self.instruction = instruction
        self.uuid_str = kwargs.get('uuid_str', None)

    def run(self, *args, **kwargs) -> Union[str, Iterator[str]]:
        if 'lang' not in kwargs:
            if has_chinese_chars([args, kwargs]):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'
        if 'uuid_str' not in kwargs and self.uuid_str is not None:
            kwargs['uuid_str'] = self.uuid_str
        return self._run(*args, **kwargs)

    @abstractmethod
    def _run(self, *args, **kwargs) -> Union[str, Iterator[str]]:
        raise NotImplementedError

    def _call_llm(self,
                  prompt: Optional[str] = None,
                  messages: Optional[List[Dict]] = None,
                  stop: Optional[List[str]] = None,
                  **kwargs) -> Union[str, Iterator[str]]:
        return self.llm.chat(
            prompt=prompt,
            messages=messages,
            stop=stop,
            stream=self.stream,
            **kwargs)

    def _call_tool(self, tool_name: str, tool_args: str, **kwargs):
        """
        Use when calling tools in bot()

        """
        return self.function_map[tool_name].call(tool_args, **kwargs)

    def _register_tool(self, tool: Union[str, Dict]):
        """
        Instantiate the global tool for the agent

        Args:
            tool: the tool should be either in a string format with name as value
            and in a dict format, example
            (1) When str: amap_weather
            (2) When dict: {'amap_weather': {'token': 'xxx'}}

        Returns:

        """
        tool_name = tool
        tool_cfg = {}
        if isinstance(tool, Dict):
            tool_name = next(iter(tool))
            tool_cfg = tool[tool_name]
        if tool_name not in TOOL_REGISTRY:
            raise NotImplementedError
        if tool not in self.function_list:
            self.function_list.append(tool)
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    def _detect_tool(self, message: Union[str, dict]):
        # use built-in default judgment functions
        if isinstance(message, str):
            return self._detect_tool_by_special_token(message)
        else:
            return self._detect_tool_by_func_call(message)

    def _detect_tool_by_special_token(self, text: str):
        """
        A built-in tool call detection: After encapsulating function calls in the LLM layer, this is no longer needed

        """
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        special_obs_token = '\nObservation:'
        func_name, func_args = None, None
        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)
        k = text.rfind(special_obs_token)
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is ommited by the LLM,
                # because the output text may have discarded the stop word.
                text = text.rstrip() + special_obs_token  # Add it back.
            k = text.rfind(special_obs_token)
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):k].strip()
            text = text[:k]  # Discard '\nObservation:'.

        return (func_name is not None), func_name, func_args, text

    def _detect_tool_by_func_call(self, message: Dict):
        """
        A built-in tool call detection for func_call format

        """
        func_name = None
        func_args = None
        if 'function_call' in message and message['function_call']:
            func_call = message['function_call']
            func_name = func_call.get('name', '')
            func_args = func_call.get('arguments', '')
        text = message['content']

        return (func_name is not None), func_name, func_args, text

    def send(self,
             message: Union[Dict, str],
             recipient: 'Agent',
             request_reply: Optional[bool] = None,
             **kwargs):
        recipient.receive(message, self, request_reply)

    def receive(self,
                message: Union[Dict, str],
                sender: 'Agent',
                request_reply: Optional[bool] = None,
                **kwargs):
        if request_reply is False or request_reply is None:
            return
        reply = self.run(message, sender=sender, **kwargs)
        if reply is not None:
            self.send(reply, sender, **kwargs)
