from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

from modelscope_agent.llm import get_chat_model
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.tools.base import (TOOL_REGISTRY, BaseTool,
                                         ToolServiceProxy)
from modelscope_agent.utils.utils import has_chinese_chars


class Agent(ABC):
    function_map: dict = {
    }  # used to record all the tools' instance, moving here to avoid `del` method crash.

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 use_tool_api: bool = False,
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
            use_tool_api: whether to use the tool service api, else to use the tool cls instance
            kwargs: other potential parameters
        """
        if isinstance(llm, Dict):
            self.llm_config = llm
            self.llm = get_chat_model(**self.llm_config)
        else:
            self.llm = llm
        self.stream = kwargs.get('stream', True)
        self.use_tool_api = use_tool_api

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

    def run(self, *args, **kwargs) -> Union[str, Iterator[str]]:
        if 'lang' not in kwargs:
            if has_chinese_chars([args, kwargs]):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'
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
        try:
            result = self.function_map[tool_name].call(tool_args, **kwargs)
        except BaseException as e:
            result = f'Tool api {tool_name} failed to call. Args: {tool_args}.'
            result += f'Details: {str(e)[:200]}'
        return result

    def _register_tool(self,
                       tool: Union[str, Dict],
                       tenant_id: str = 'default'):
        """
        Instantiate the tool for the agent

        Args:
            tool: the tool should be either in a string format with name as value
            and in a dict format, example
            (1) When str: amap_weather
            (2) When dict: {'amap_weather': {'token': 'xxx'}}
            tenant_id: the tenant_id of the tool is now  for code interpreter that need to keep track of the tenant
        Returns:

        """
        tool_name = tool
        tool_cfg = {}
        if isinstance(tool, dict):
            tool_name = next(iter(tool))
            tool_cfg = tool[tool_name]
        if tool_name not in TOOL_REGISTRY:
            raise NotImplementedError
        if tool not in self.function_list:
            self.function_list.append(tool)

            tool_class_with_tenant = TOOL_REGISTRY[tool_name]

            # adapt the TOOL_REGISTRY[tool_name] to origin tool class

            if isinstance(tool_class_with_tenant, BaseTool):
                tool_class_with_tenant = {'class': TOOL_REGISTRY[tool_name]}
                TOOL_REGISTRY[tool_name] = tool_class_with_tenant

            # check if the tenant_id of tool instance or tool service are exists
            # TODO: change from use_tool_api=True to False, to get the tenant_id of the tool changes to
            if tenant_id in tool_class_with_tenant and self.use_tool_api:
                return

            try:
                if self.use_tool_api:
                    # get service proxy as tool instance, call method will call remote tool service
                    tool_instance = ToolServiceProxy(tool_name, tool_cfg,
                                                     tenant_id)
                else:
                    # instantiation tool class as tool instance
                    tool_instance = TOOL_REGISTRY[tool_name]['class'](tool_cfg)

                self.function_map[tool_name] = tool_instance

            except TypeError:
                # When using OpenAPI, tool_class is already an instantiated object, not a corresponding class
                self.function_map[tool_name] = TOOL_REGISTRY[tool_name][
                    'class']
            except Exception as e:
                raise RuntimeError(e)

            # store the instance of tenant to tool registry on this tool
            tool_class_with_tenant[tenant_id] = self.function_map[tool_name]

    def _detect_tool(self, message: Union[str,
                                          dict]) -> Tuple[bool, str, str, str]:
        """
        A built-in tool call detection for func_call format

        Args:
            message: one message
                (1) When dict: Determine whether to call the tool through the function call format.
                (2) When str: The tool needs to be parsed from the string, and at this point, the agent subclass needs
                              to implement a custom _detect_tool function.

        Returns:
            - bool: need to call tool or not
            - str: tool name
            - str: tool args
            - str: text replies except for tool calls
        """
        func_name = None
        func_args = None
        assert isinstance(message, dict)
        if 'function_call' in message and message['function_call']:
            func_call = message['function_call']
            func_name = func_call.get('name', '')
            func_args = func_call.get('arguments', '')
        # Compatible with OpenAI API
        if 'tool_calls' in message and message['tool_calls']:
            func_call = message['tool_calls'][0]['function']
            func_name = func_call.get('name', '')
            func_args = func_call.get('arguments', '')

        text = message.get('content', '')

        return (func_name is not None), func_name, func_args, text

    def _parse_image_url(self, image_url: List[Union[str, Dict]],
                         messages: List[Dict]) -> List[Dict]:

        assert len(messages) > 0

        if isinstance(image_url[0], str):
            image_url = [{'url': url} for url in image_url]

        images = [{
            'type': 'image_url',
            'image_url': image
        } for image in image_url]

        origin_message: str = messages[-1]['content']
        parsed_message = [{'type': 'text', 'text': origin_message}, *images]
        messages[-1]['content'] = parsed_message
        return messages

    # del the tools as well while del the agent
    def __del__(self):
        try:
            for tool_instance in self.function_map.items():
                del tool_instance
        except Exception:
            pass
