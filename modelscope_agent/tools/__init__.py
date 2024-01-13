from .amap_weather import AMAPWeather
from .base import TOOL_REGISTRY, BaseTool, register_tool
from .code_interpreter import CodeInterpreter
from .dashscope_tools.image_generation import TextToImageTool
from .dashscope_tools.phantom_tool import Phantom
from .dashscope_tools.qwen_vl import QWenVL
from .dashscope_tools.style_repaint import StyleRepaint
from .dashscope_tools.wordart_tool import WordArtTexture
from .doc_parser import DocParser
from .langchain_proxy_tool import LangchainTool
from .modelscope_tools.pipeline_tool import ModelscopePipelineTool
from .modelscope_tools.text_to_speech_tool import TexttoSpeechTool
from .modelscope_tools.text_to_video_tool import TextToVideoTool
from .openapi_plugin import OpenAPIPluginTool
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
