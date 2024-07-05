import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import json
import json5
import requests
from modelscope_agent.constants import (BASE64_FILES,
                                        DEFAULT_CODE_INTERPRETER_DIR,
                                        DEFAULT_TOOL_MANAGER_SERVICE_URL,
                                        LOCAL_FILE_PATHS,
                                        MODELSCOPE_AGENT_TOKEN_HEADER_NAME)
from modelscope_agent.utils.base64_utils import decode_base64_to_files
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.utils import has_chinese_chars

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', DEFAULT_CODE_INTERPRETER_DIR)

# ast?
register_map = {
    'amap_weather':
    'AMAPWeather',
    'storage':
    'Storage',
    'web_search':
    'WebSearch',
    'image_gen':
    'TextToImageTool',
    'image_gen_lite':
    'TextToImageLiteTool',
    'image_enhancement':
    'ImageEnhancement',
    'qwen_vl':
    'QWenVL',
    'style_repaint':
    'StyleRepaint',
    'paraformer_asr':
    'ParaformerAsrTool',
    'sambert_tts':
    'SambertTtsTool',
    'wordart_texture_generation':
    'WordArtTexture',
    'detect_for_google_translate':
    'DetectForGoogleTranslate',
    'languages_for_google_translate':
    'LanguagesForGoogleTranslate',
    'translate_for_google_translate':
    'TranslateForGoogleTranslate',
    'get_data_fact_for_numbers':
    'GetDataFactForNumbers',
    'get_math_fact_for_numbers':
    'GetMathFactForNumbers',
    'get_year_fact_for_numbers':
    'GetYearFactForNumbers',
    'model_scope_text_ie':
    'TextinfoextracttoolForAlphaUmi',
    'search_torrents_for_movie_tv_music_search_and_download':
    'SearchTorrentsForMovieTvMusicSearchAndDownload',
    'get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download':
    'GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload',
    'get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download':
    'GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload',
    'get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download':
    'GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload',
    'get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download':
    'GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload',
    'listquotes_for_current_exchange':
    'ListquotesForCurrentExchange',
    'exchange_for_current_exchange':
    'exchange_for_current_exchange',
    'similarity_search':
    'SimilaritySearch',
    'hf-tool':
    'HFTool',
    'RenewInstance':
    'AliyunRenewInstanceTool',
    'web_browser':
    'WebBrowser',
    'text-address':
    'TextAddressTool',
    'image-chat':
    'ImageChatTool',
    'text-ner':
    'TextNerTool',
    'speech-generation':
    'TexttoSpeechTool',
    'video-generation':
    'TextToVideoTool',
    'text-ie':
    'TextInfoExtractTool',
    'openapi_plugin':
    'OpenAPIPluginTool',
    'langchain_tool':
    'LangchainTool',
    'code_interpreter':
    'CodeInterpreter',
    'doc_parser':
    'DocParser'
}


def import_from_register(key):
    value = register_map[key]
    exec(f'from . import {value}')


class ToolRegistry(dict):

    def _import_key(self, key):
        try:
            import_from_register(key)
        except Exception as e:
            print(f'import {key} failed, details: {e}')

    def __getitem__(self, key):
        if key not in self.keys():
            self._import_key(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._import_key(key)
        return super().__contains__(key)


TOOL_REGISTRY = ToolRegistry()


def register_tool(name):

    def decorator(cls):
        TOOL_REGISTRY[name] = {'class': cls}
        return cls

    return decorator


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
        except Exception as e:
            print(e)
            params = params.replace('\r', '\\r').replace('\n', '\\n')
            params_json = json5.loads(params)

        for param in self.parameters:
            if 'required' in param and param['required']:
                if param['name'] not in params_json:
                    raise ValueError(f'param `{param["name"]}` is required')
        return params_json

    def _parse_files_input(self, *args, **kwargs):
        # convert image_file_paths from string to list
        if BASE64_FILES in kwargs:
            # if image_file_paths is base64
            base64_files = kwargs.pop(BASE64_FILES)
            local_file_paths = decode_base64_to_files(base64_files, WORK_DIR)
            kwargs[LOCAL_FILE_PATHS] = local_file_paths
        return kwargs

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

    @staticmethod
    def parser_function(tools: List[dict]):
        """
        Tool parser function with input
        Args:
            tools: the list of tools, each tool includes key of name, description, parameters

        Returns:

        """

        tool_desc_template = {
            'zh':
            '{name}: {name} API。{description} 输入参数: {parameters} Format the arguments as a JSON object.',
            'en':
            '{name}: {name} API. {description} Parameters: {parameters} Format the arguments as a JSON object.'
        }

        def has_chinese_chars_in_tools(_tools):
            for _tool in _tools:
                _func_info = _tool.get('function', {})
                if 'description' in _func_info:
                    if has_chinese_chars(_func_info['description']):
                        return True

            return False

        if has_chinese_chars_in_tools(tools):
            tool_desc = tool_desc_template['zh']
        else:
            tool_desc = tool_desc_template['en']

        tools_text = []
        for tool in tools:
            func_info = tool.get('function', {})
            if func_info == {}:
                continue
            name = func_info.get('name', '')
            description = func_info.get('description', '')
            tool = tool_desc.format(
                name=name,
                description=description,
                parameters=json.dumps(
                    func_info['parameters'], ensure_ascii=False),
            )
            tools_text.append(tool)
        tools_text = '\n\n'.join(tools_text)
        return tools_text


class ToolServiceProxy(BaseTool):

    def __init__(self,
                 tool_name: str,
                 tool_cfg: dict,
                 tenant_id: str = 'default',
                 tool_service_manager_url: str = os.getenv(
                     'TOOL_MANAGER_SERVICE_URL',
                     DEFAULT_TOOL_MANAGER_SERVICE_URL),
                 user_token: str = None,
                 **kwargs):
        """
        Tool service proxy class
        Args:
            tool_name: tool name might be the name of tool or the address of tool artifacts
            tool_cfg: the configuration of tool
            tenant_id: the tenant id that the tool belongs to, defalut to 'default'
            tool_service_manager_url: the url of tool service manager, default to 'http://localhost:31511'
            user_token: used to pass to the tool service manager to authenticate the user
        """

        self.tool_service_manager_url = tool_service_manager_url
        self.user_token = user_token
        self.tool_name = tool_name
        self.tool_cfg = tool_cfg
        self.tenant_id = tenant_id
        self._register_tool()

        max_retry = 10
        while max_retry > 0:
            status = self._check_tool_status()
            if status == 'running':
                break
            time.sleep(1)
            max_retry -= 1
        if max_retry == 0:
            raise RuntimeError(
                'Tool node not start up successfully, please double check your docker environment'
            )

        tool_info = self._get_tool_info()
        self.name = tool_info['name']
        self.description = tool_info['description']
        self.parameters = tool_info['parameters']
        super().__init__({self.name: self.tool_cfg})

    @staticmethod
    def parse_service_response(response):
        try:
            # Assuming the response is a JSON string
            response_data = response.json()

            # Extract the 'output' field from the response
            output_data = response_data.get('output', {})
            return output_data
        except json.JSONDecodeError:
            # Handle the case where response is not JSON or cannot be decoded
            return None

    def _register_tool(self):
        try:
            service_token = os.getenv('TOOL_MANAGER_AUTH', '')
            headers = {
                'Content-Type': 'application/json',
                MODELSCOPE_AGENT_TOKEN_HEADER_NAME: self.user_token,
                'authorization': service_token
            }
            print(f'reach here create {headers}')
            response = requests.post(
                f'{self.tool_service_manager_url}/create_tool_service',
                json={
                    'tool_name': self.tool_name,
                    'tenant_id': self.tenant_id,
                    'tool_cfg': self.tool_cfg
                },
                headers=headers)
            response.raise_for_status()
            result = ToolServiceProxy.parse_service_response(response)
            if 'status' not in result:
                raise Exception(
                    'Failed to register tool, the tool service might be done, please use local version'
                )
            if result['status'] not in ['pending', 'running']:
                raise Exception(
                    'Failed to register tool, the tool service might be done, please use local version.'
                )
        except Exception as e:
            raise RuntimeError(
                f'Get error during registering tool from tool manager service with detail {e}.'
            )

    def _check_tool_status(self):
        try:
            service_token = os.getenv('TOOL_MANAGER_AUTH', '')
            headers = {
                'Content-Type': 'application/json',
                MODELSCOPE_AGENT_TOKEN_HEADER_NAME: self.user_token,
                'authorization': service_token
            }
            response = requests.get(
                f'{self.tool_service_manager_url}/check_tool_service_status',
                params={
                    'tool_name': self.tool_name,
                    'tenant_id': self.tenant_id,
                },
                headers=headers)
            response.raise_for_status()
            result = ToolServiceProxy.parse_service_response(response)
            if 'status' not in result:
                raise Exception(
                    'Failed to register tool, the tool service might be done, please use local version'
                )
            return result['status']
        except Exception as e:
            raise RuntimeError(
                f'Get error during checking status from tool manager service with detail {e}'
            )

    def _get_tool_info(self):
        try:
            service_token = os.getenv('TOOL_MANAGER_AUTH', '')
            headers = {
                'Content-Type': 'application/json',
                MODELSCOPE_AGENT_TOKEN_HEADER_NAME: self.user_token,
                'authorization': service_token
            }
            logger.query_info(message=f'tool_info requests header {headers}')
            response = requests.post(
                f'{self.tool_service_manager_url}/tool_info',
                json={
                    'tool_name': self.tool_name,
                    'tenant_id': self.tenant_id
                },
                headers=headers)
            response.raise_for_status()
            return ToolServiceProxy.parse_service_response(response)
        except Exception as e:
            raise RuntimeError(
                f'Get error during getting tool info from tool manager service with detail {e}'
            )

    def call(self, params: str, **kwargs):
        # ms_token
        self.user_token = kwargs.get('user_token', self.user_token)
        service_token = os.getenv('TOOL_MANAGER_AUTH', '')
        headers = {
            'Content-Type': 'application/json',
            MODELSCOPE_AGENT_TOKEN_HEADER_NAME: self.user_token,
            'authorization': service_token
        }
        logger.query_info(message=f'calling tool header {headers}')

        try:
            # visit tool node to call tool
            response = requests.post(
                f'{self.tool_service_manager_url}/execute_tool',
                json={
                    'tool_name': self.tool_name,
                    'tenant_id': self.tenant_id,
                    'params': params,
                    'kwargs': kwargs
                },
                headers=headers)
            logger.query_info(
                message=f'calling tool message {response.json()}')

            response.raise_for_status()
            return ToolServiceProxy.parse_service_response(response)
        except Exception as e:
            raise RuntimeError(
                f'Get error during executing tool from tool manager service with detail {e}'
            )
