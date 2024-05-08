from collections import defaultdict
from typing import Union

from modelscope_agent.tools.base import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


@register_tool('text-ie')
class TextInfoExtractTool(ModelscopePipelineTool):
    default_model = 'damo/nlp_structbert_siamese-uie_chinese-base'
    description = '信息抽取服务，针对中文的文本，根据schema要抽取的内容，找出其中对应信息，并用json格式展示'
    name = 'text-ie'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的文本',
        'required': True,
        'type': 'string'
    }, {
        'name': 'schema',
        'description': '要抽取信息的json表示',
        'required': True,
        'type': 'dict'
    }]
    task = Tasks.siamese_uie
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/nlp_structbert_siamese-uie_chinese-base'

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        InfoExtract = defaultdict(list)
        for e in result['Data']['output']:
            InfoExtract[e[0]['type']].append(e[0]['span'])
        return str(dict(InfoExtract))

    def _verify_args(self, params: str) -> Union[str, dict]:
        params = super()._verify_args(params)
        params['parameters'] = {'schema': params['schema']}
        params.pop('schema')
        return params
