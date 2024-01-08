from collections import defaultdict

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class TextInfoExtractTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_structbert_siamese-uie_chinese-base'
    description = '信息抽取服务，针对中文的文本，根据schema要抽取的内容，找出其中对应信息，并用json格式展示'
    name = 'text-ie'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的文本',
        'required': True
    }, {
        'name': 'schema',
        'description': '要抽取信息的json表示',
        'required': True
    }]
    task = Tasks.siamese_uie

    def _remote_parse_input(self, *args, **kwargs):
        kwargs['parameters'] = {'schema': kwargs['schema']}
        kwargs.pop('schema')
        return kwargs

    def _parse_output(self, origin_result, *args, **kwargs):
        final_result = defaultdict(list)
        for e in origin_result['output']:
            final_result[e[0]['type']].append(e[0]['span'])

        return {'result': dict(final_result)}
