from collections import defaultdict

from modelscope_agent.tools.base import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


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
        ner = defaultdict(list)
        for e in result['Data']['output']:
            ner[e['type']].append(e['span'])
        return str(dict(ner))
