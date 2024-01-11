from modelscope_agent.tools import register_tool

from .pipeline_tool import ModelscopePipelineTool


@register_tool('text-translation-en2zh')
class TranslationEn2ZhTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_csanmt_translation_en2zh'
    description = '根据输入指令，将相应的英文文本翻译成中文回复'
    name = 'text-translation-en2zh'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的英文文本',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        zh = result['Data']['translation']
        return zh
