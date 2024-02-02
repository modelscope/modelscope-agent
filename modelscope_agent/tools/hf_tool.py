from modelscope_agent.tools.base import BaseTool, register_tool
from transformers.tools import Tool as HFTool
from copy import deepcopy

@register_tool('HFTool')
class HFTool(BaseTool):
    description = '通过调用langchain插件来支持对语言模型的输入输出格式进行处理，输入文本字符，输出经过格式处理的结果'
    name = 'HFTool'
    parameters: list = [{
        'name': 'commands',
        'description': '需要进行格式处理的文本字符列表',
        'required': True,
        'type': 'string'
    }]
    def __init__(self, HFTool):
        from transformers.tools import Tool as HF_Tool
        if not isinstance(HFTool, HF_Tool):
            raise ValueError('HFTool should be type of HF tool')
        self.HFTool = HFTool
    
    def parse_langchain_schema(self):
        self.description = self.HFTool.description
        self.name = self.HFTool.name
        self.parameters = []
        for name, arg in self.HFTool.args.items():
            tool_arg = deepcopy(arg)
            tool_arg['name'] = name
            tool_arg['required'] = True
            tool_arg['type'] = arg['anyOf'][0].get('type', 'string')
            tool_arg.pop('title')
            self.parameters.append(tool_arg)
    
    def call(self, params: str, **kwargs):
        params = self._verify_args(params)
        res = self.HFTool(params)
        return res