from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import json
import json5
import requests
from modelscope_agent.utils.utils import has_chinese_chars

TOOL_REGISTRY = {}


def register_tool(name):

    def decorator(cls):
        if name not in TOOL_REGISTRY:
            TOOL_REGISTRY[name] = {}
        TOOL_REGISTRY[name]['class'] = cls
        return cls

    return decorator


class ToolServiceProxy:

    def __init__(
        self,
        tool_name: str,
        tool_cfg: dict,
        tenant_id: str = 'default',
    ):
        """
        Tool service proxy class
        Args:
            tool_name: tool name might be the name of tool or the address of tool artifacts
            tool_cfg:
            tenant_id:
        """
        self.tool_service_manager_url = 'http://tool-service-manager:8000'
        self.tool_name = tool_name
        self.tool_cfg = tool_cfg
        self.tenant_id = tenant_id
        self._register_tool()

    def _register_tool(self):
        response = requests.post(
            f'{self.tool_service_manager_url}/create_tool_service',
            params={
                'tool_name': self.tool_name,
                'tenant_id': self.tenant_id,
                'tool_cfg': self.tool_cfg
            })
        response.raise_for_status()
        result = response.json()
        if 'tool_node_name' not in result:
            raise Exception(
                'Failed to register tool, the tool service might be done, please use local version'
            )

    def _get_tool_api_endpoint(self):
        # get tool node endpoint by tool service
        response = requests.get(
            f'{self.tool_service_manager_url}/get_tool_service_url',
            params={
                'tool_name': self.tool_name,
                'tenant_id': self.tenant_id
            })
        response.raise_for_status()
        tool_info = response.json()
        return tool_info['api_endpoint']

    def call(self, params: str, **kwargs):
        # visit tool node to call tool
        api_endpoint = self._get_tool_api_endpoint()
        response = requests.post(api_endpoint, json={'params': params})
        response.raise_for_status()
        return response.json()


class BaseTool(ABC):
    name: str
    description: str
    parameters: List[Dict]

    def __init__(self, cfg: Optional[Dict] = {}):
        """
        :param schema: Format of tools, default to oai format, in case there is a need for other formats
        """
        self.cfg = cfg.get(self.name, {})

        self.schema = self.cfg.get('schema', 'oai')
        self.function = self._build_function()
        self.function_plain_text = self._parser_function()

    @abstractmethod
    def call(self, params: str, **kwargs):
        """
        The interface for calling tools

        :param params: the parameters of func_call
        :param kwargs: additional parameters for calling tools
        :return: the result returned by the tool, implemented in the subclass
        """
        raise NotImplementedError

    def _verify_args(self, params: str) -> Union[str, dict]:
        """
        Verify the parameters of the function call

        :param params: the parameters of func_call
        :return: the str params or the legal dict params
        """
        try:
            params_json = json5.loads(params)
            for param in self.parameters:
                if 'required' in param and param['required']:
                    if param['name'] not in params_json:
                        return params
            return params_json
        except Exception:
            return params

    def _build_function(self):
        """
        The dict format after applying the template to the function, such as oai format

        """
        if self.schema == 'oai':
            function = {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': [],
                },
            }
            for para in self.parameters:
                function_details = {
                    'type':
                    para['type'] if 'type' in para else para['schema']['type'],
                    'description':
                    para['description']
                }
                if 'enum' in para and para['enum'] not in ['', []]:
                    function_details['enum'] = para['enum']
                function['parameters']['properties'][
                    para['name']] = function_details
                if 'required' in para and para['required']:
                    function['parameters']['required'].append(para['name'])
        else:
            function = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }

        return function

    def _parser_function(self):
        """
        Text description of function

        """
        tool_desc_template = {
            'zh':
            '{name}: {name} API。{description} 输入参数: {parameters} Format the arguments as a JSON object.',
            'en':
            '{name}: {name} API. {description} Parameters: {parameters} Format the arguments as a JSON object.'
        }

        if has_chinese_chars(self.function['description']):
            tool_desc = tool_desc_template['zh']
        else:
            tool_desc = tool_desc_template['en']

        return tool_desc.format(
            name=self.function['name'],
            description=self.function['description'],
            parameters=json.dumps(
                self.function['parameters'], ensure_ascii=False),
        )
