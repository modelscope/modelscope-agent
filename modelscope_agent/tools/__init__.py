import sys

from ..utils import _LazyModule
from .contrib import *  # noqa F403

_import_structure = {
    'amap_weather': ['AMAPWeather'],
    'code_interpreter': ['CodeInterpreter'],
    'contrib': ['AliyunRenewInstanceTool'],
    'dashscope_tools': [
        'ImageEnhancement', 'TextToImageTool', 'TextToImageLiteTool',
        'ParaformerAsrTool', 'QWenVL', 'SambertTtsTool', 'StyleRepaint',
        'WordArtTexture'
    ],
    'doc_parser': ['DocParser'],
    'hf_tool': ['HFTool'],
    'langchain_proxy_tool': ['LangchainTool'],
    'modelscope_tools': [
        'ImageChatTool', 'ModelscopePipelineTool', 'TextAddressTool',
        'TextInfoExtractTool', 'TextNerTool', 'TexttoSpeechTool',
        'TextToVideoTool', 'TranslationEn2ZhTool', 'TranslationZh2EnTool'
    ],
    'openapi_plugin': ['OpenAPIPluginTool'],
    'rapidapi_tools': [
        'ListquotesForCurrentExchange', 'exchange_for_current_exchange',
        'TextinfoextracttoolForAlphaUmi',
        'GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload',
        'SearchTorrentsForMovieTvMusicSearchAndDownload',
        'GetDataFactForNumbers,', 'GetMathFactForNumbers',
        'GetYearFactForNumbers', 'DetectForGoogleTranslate',
        'LanguagesForGoogleTranslate', 'TranslateForGoogleTranslate'
    ],
    'similarity_search': ['SimilaritySearch'],
    'storage_proxy_tool': ['Storage'],
    'web_browser': ['WebBrowser'],
    'web_search': ['WebSearch'],
    'base': ['TOOL_REGISTRY', 'BaseTool', 'register_tool', 'ToolServiceProxy'],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
