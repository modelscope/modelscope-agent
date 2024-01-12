from modelscope_agent.tools import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool

import json

@register_tool('text-ner')
class TextNerTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_raner_named-entity-recognition_chinese-base-cmeee'
    description = '命名实体识别服务，针对需要识别的中文文本，找出其中的实体，返回json格式结果'
    name = 'text-ner'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的文本',
        'required': True,
        'type': 'string'
    }]
    task = Tasks.named_entity_recognition
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/nlp_raner_named-entity-recognition_chinese-base-cmeee'  # noqa E501

    def _remote_call(self, params: str, **kwargs) -> str:
        result = super()._remote_call(params, **kwargs)
        ner = result['Data']['output']
        return str(ner)

    def _local_call(self, params: dict, **kwargs) -> str:
        result = super()._local_call(params, **kwargs)
        result = json.loads(result)
        ner = result['output']
        return str(ner)