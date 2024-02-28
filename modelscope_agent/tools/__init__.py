from .amap_weather import AMAPWeather
from .base import TOOL_REGISTRY, BaseTool, register_tool
from .code_interpreter import CodeInterpreter
from .dashscope_tools.image_enhancement import ImageEnhancement
from .dashscope_tools.image_generation import TextToImageTool
from .dashscope_tools.paraformer_asr_tool import ParaformerAsrTool
from .dashscope_tools.qwen_vl import QWenVL
from .dashscope_tools.sambert_tts_tool import SambertTtsTool
from .dashscope_tools.style_repaint import StyleRepaint
from .dashscope_tools.wordart_tool import WordArtTexture
from .doc_parser import DocParser
from .hf_tool import HFTool
from .langchain_proxy_tool import LangchainTool
from .modelscope_tools.image_chat_tool import ImageChatTool
from .modelscope_tools.pipeline_tool import ModelscopePipelineTool
from .modelscope_tools.text_address_tool import TextAddressTool
from .modelscope_tools.text_ie_tool import TextInfoExtractTool
from .modelscope_tools.text_ner_tool import TextNerTool
from .modelscope_tools.text_to_speech_tool import TexttoSpeechTool
from .modelscope_tools.text_to_video_tool import TextToVideoTool
from .modelscope_tools.translation_en2zh_tool import TranslationEn2ZhTool
from .modelscope_tools.translation_zh2en_tool import TranslationZh2EnTool
from .openapi_plugin import OpenAPIPluginTool
from .rapidapi_tools.Finance.current_exchage import (
    exchange_for_current_exchange, listquotes_for_current_exchange)
from .rapidapi_tools.Modelscope.text_ie_tool import \
    TextInfoExtractTool_for_alpha_umi
from .rapidapi_tools.Movies.movie_tv_music_search_and_download import (
    get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download,
    get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download,
    get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download,
    get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download,
    search_torrents_for_movie_tv_music_search_and_download)
from .rapidapi_tools.Number.numbers import (get_data_fact_for_numbers,
                                            get_math_fact_for_numbers,
                                            get_year_fact_for_numbers)
from .rapidapi_tools.Translate.google_translate import (
    detect_for_google_translate, languages_for_google_translate,
    translate_for_google_translate)
from .similarity_search import SimilaritySearch
from .storage_proxy_tool import Storage
from .web_browser import WebBrowser
from .web_search import WebSearch


def call_tool(plugin_name: str, plugin_args: str) -> str:
    if plugin_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[plugin_name].call(plugin_args)
    else:
        raise NotImplementedError


__all__ = ['BaseTool', 'TOOL_REGISTRY', 'register_tool']
