from .amap_weather import AMAPWeather
from .code_interperter import CodeInterpreter
from .code_interpreter_jupyter import CodeInterpreterJupyter
from .hf_tool import HFTool
from .image_chat_tool import ImageChatTool
from .pipeline_tool import ModelscopePipelineTool
from .plugin_tool import LangchainTool
from .text_address_tool import TextAddressTool
from .text_ie_tool import TextInfoExtractTool
from .text_ner_tool import TextNerTool
from .text_to_image_tool import TextToImageTool
from .text_to_speech_tool import TexttoSpeechTool
from .text_to_video_tool import TextToVideoTool
from .tool import Tool
from .translation_en2zh_tool import TranslationEn2ZhTool
from .translation_zh2en_tool import TranslationZh2EnTool
from .wordart_tool import WordArtTexture
from .websearch import WebSearch

TOOL_INFO_LIST = {
    'modelscope_text-translation-zh2en': 'TranslationZh2EnTool',
    'modelscope_text-translation-en2zh': 'TranslationEn2ZhTool',
    'modelscope_text-ie': 'TextInfoExtractTool',
    'modelscope_text-ner': 'TextNerTool',
    'modelscope_text-address': 'TextAddressTool',
    'image_gen': 'TextToImageTool',
    'modelscope_video-generation': 'TextToVideoTool',
    'modelscope_image-chat': 'ImageChatTool',
    'modelscope_speech-generation': 'TexttoSpeechTool',
    'amap_weather': 'AMAPWeather',
    'code_interpreter': 'CodeInterpreterJupyter',
    'wordart_texture_generation': 'WordArtTexture',
    'web_search': 'WebSearch'
}
