from .custom_tool import AliyunRenewInstanceTool
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

DEFAULT_TOOL_LIST = {
    'modelscope_text-translation-zh2en': 'TranslationZh2EnTool',
    'modelscope_text-translation-en2zh': 'TranslationEn2ZhTool',
    'modelscope_text-ie': 'TextInfoExtractTool',
    'modelscope_text-ner': 'TextNerTool',
    'modelscope_text-address': 'TextAddressTool',
    'modelscope_image-generation': 'TextToImageTool',
    'modelscope_video-generation': 'TextToVideoTool',
    'modelscope_image-chat': 'ImageChatTool',
    'modelscope_speech-generation': 'TexttoSpeechTool',
}
