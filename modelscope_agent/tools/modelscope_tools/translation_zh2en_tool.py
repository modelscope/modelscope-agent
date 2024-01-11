from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class TranslationZh2EnTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_csanmt_translation_zh2en'
    description = '根据输入指令，将相应的中文文本翻译成英文回复'
    name = 'text-translation-zh2en'
    task = Tasks.translation
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的中文文本',
        'required': True
    }]

    def _parse_output(self, origin_result, *args, **kwargs):
        return {'result': origin_result['translation']}
