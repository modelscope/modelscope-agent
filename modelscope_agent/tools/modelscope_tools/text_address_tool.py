import json
from modelscope_agent.tools import register_tool

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
    url = ('https://api-inference.modelscope.cn/api-inference/v1/models/'
           'damo/mgeo_geographic_elements_tagging_chinese_base')

    def _remote_call(self, params: str, **kwargs) -> str:
        result = super()._remote_call(params, **kwargs)
        address = result['Data']['output']
        return str(address)

    def _local_call(self, params: dict, **kwargs) -> str:
        result = super()._local_call(params, **kwargs)
        result = json.loads(result)
        address = result['output']
        return str(address)
