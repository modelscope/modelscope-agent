from .amap_weather import AMAPWeather
from .code_interperter import CodeInterpreter
from .code_interpreter_jupyter import CodeInterpreterJupyter
from .hf_tool import HFTool
from .image_chat_tool import ImageChatTool
from .pipeline_tool import ModelscopePipelineTool
from .plugin_tool import LangchainTool
from .qwen_vl import QWenVL
from .style_repaint import StyleRepaint
from .text_address_tool import TextAddressTool
from .text_ie_tool import TextInfoExtractTool
from .text_ner_tool import TextNerTool
from .text_to_image_tool import TextToImageTool
from .text_to_speech_tool import TexttoSpeechTool
from .text_to_video_tool import TextToVideoTool
from .tool import Tool
from .translation_en2zh_tool import TranslationEn2ZhTool
from .translation_zh2en_tool import TranslationZh2EnTool
from .web_browser import WebBrowser
from .web_search import WebSearch
from .wordart_tool import WordArtTexture

TOOL_INFO_LIST = {
    'text-translation-zh2en': 'TranslationZh2EnTool',
    'text-translation-en2zh': 'TranslationEn2ZhTool',
    'text-ie': 'TextInfoExtractTool',
    'text-ner': 'TextNerTool',
    'text-address': 'TextAddressTool',
    'image_gen': 'TextToImageTool',
    'video-generation': 'TextToVideoTool',
    'image-chat': 'ImageChatTool',
    'speech-generation': 'TexttoSpeechTool',
    'amap_weather': 'AMAPWeather',
    'code_interpreter': 'CodeInterpreterJupyter',
    'wordart_texture_generation': 'WordArtTexture',
    'web_search': 'WebSearch',
    'web_browser': 'WebBrowser',
    'qwen_vl': 'QWenVL',
    'style_repaint': 'StyleRepaint',
}
