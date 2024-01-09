from .amap_weather import AMAPWeather
from .base import TOOL_REGISTRY, BaseTool
from .code_interpreter_jupyter import CodeInterpreterJupyter
from .dashscope_tools.image_generation import TextToImageTool
from .dashscope_tools.style_repaint import StyleRepaint
from .langchain_tool import LangchainTool
from .openapi_plugin import OpenAPIPluginTool


def call_tool(plugin_name: str, plugin_args: str) -> str:
    if plugin_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[plugin_name].call(plugin_args)
    else:
        raise NotImplementedError


__all__ = ['BaseTool', 'TOOL_REGISTRY']
