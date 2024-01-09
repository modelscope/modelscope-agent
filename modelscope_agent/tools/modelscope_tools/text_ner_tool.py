from collections import defaultdict

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class TextNerTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_raner_named-entity-recognition_chinese-base-news'
    description = '命名实体识别服务，针对需要识别的中文文本，找出其中的实体，返回json格式结果'
    name = 'text-ner'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的文本',
        'required': True
    }]
    task = Tasks.named_entity_recognition

    def _parse_output(self, origin_result, *args, **kwargs):
        final_result = defaultdict(list)
        for e in origin_result['output']:
            final_result[e['type']].append(e['span'])
        return {'result': dict(final_result)}
