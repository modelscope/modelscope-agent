import json
from modelscope_agent.tools.base import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


@register_tool('text-address')
class TextAddressTool(ModelscopePipelineTool):
    default_model = 'damo/mgeo_geographic_elements_tagging_chinese_base'
    description = '地址解析服务，针对中文地址信息，识别出里面的元素，包括省、市、区、镇、社区、道路、路号、POI、楼栋号、户室号等'
    name = 'text-address'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的地址信息',
        'required': True,
        'type': 'string'
    }]
    task = Tasks.token_classification
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/mgeo_geographic_elements_tagging_chinese_base'  # noqa E501

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        address = {}
        for e in result['Data']['output']:
            address[e['type']] = e['span']
        return json.dumps(address, ensure_ascii=False)
