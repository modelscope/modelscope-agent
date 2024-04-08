import sys

from modelscope_agent.utils import _LazyModule

_import_structure = {
    'image_chat_tool': ['ImageChatTool'],
    'text_address_tool': ['TextAddressTool'],
    'text_ie_tool': ['TextInfoExtractTool'],
    'text_ner_tool': ['TextNerTool'],
    'text_to_speech_tool': ['TexttoSpeechTool'],
    'text_to_video_tool': ['TextToVideoTool'],
    'translation_en2zh_tool': ['TranslationEn2ZhTool'],
    'translation_zh2en_tool': ['TranslationZh2EnTool'],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
