import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import json
import json5
from modelscope_agent.utils.utils import has_chinese_chars

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
        TOOL_REGISTRY[name] = cls
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
        except Exception:
            params = params.replace('\r', '\\r').replace('\n', '\\n')
            params_json = json5.loads(params)

        for param in self.parameters:
            if 'required' in param and param['required']:
                if param['name'] not in params_json:
                    raise ValueError(f'param `{param["name"]}` is required')
        return params_json

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
