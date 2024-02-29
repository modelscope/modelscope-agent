from collections import defaultdict
from typing import Union

from modelscope_agent.tools import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopepipelinetoolForAlphaUmi


@register_tool('model_scope_text_ie')
class TextinfoextracttoolForAlphaUmi(ModelscopepipelinetoolForAlphaUmi):
    default_model = 'damo/nlp_structbert_siamese-uie_chinese-base'
    description = 'Information extraction service for Chinese text, which extracts specific content according to a predefined schema, \
    identifies the corresponding information, and displays it in JSON format.'

    name = 'model_scope_text_ie'
    parameters: list = [{
        'name': 'input',
        'description': 'text input by user',
        'required': True,
        'type': 'string'
    }, {
        'name': 'schema',
        'description': 'a json schema for the extrated information',
        'required': True,
        'type': 'dict'
    }]
    task = Tasks.siamese_uie
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/nlp_structbert_siamese-uie_chinese-base'

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        print(result)
        InfoExtract = defaultdict(list)
        for e in result['Data']['output']:
            InfoExtract[e[0]['type']].append(e[0]['span'])
        return str(dict(InfoExtract))

    def _verify_args(self, params: str) -> Union[str, dict]:
        params = super()._verify_args(params)
        params['parameters'] = {'schema': params['schema']}
        params.pop('schema')
        return params
