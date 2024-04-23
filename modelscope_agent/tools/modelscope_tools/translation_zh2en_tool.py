from modelscope_agent.tools.base import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


@register_tool('text-translation-zh2en')
class TranslationZh2EnTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_csanmt_translation_zh2en'
    description = '根据输入指令，将相应的中文文本翻译成英文回复'
    name = 'text-translation-zh2en'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的中文文本',
        'required': True,
        'type': 'string'
    }]
    task = Tasks.translation
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/nlp_csanmt_translation_zh2en'

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        en = result['Data']['translation']
        return en
