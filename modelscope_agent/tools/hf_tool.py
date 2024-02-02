from modelscope_agent.tools.base import BaseTool, register_tool
from transformers.tools import Tool as HFTool
import json

@register_tool('HFTool')
class HFTool(BaseTool):
    description = '通过调用HFTool插件来支持文本，音频，图片等的生成'
    name = 'HFTool'
    parameters: list = []
    def __init__(self, HFTool):
        from transformers.tools import Tool as HF_Tool
        if not isinstance(HFTool, HF_Tool):
            raise ValueError('HFTool should be type of HF tool')
        self.HFTool = HFTool
        super().__init__()
    
    def call(self, params: str, **kwargs):
        params = self._verify_args(params)
        res = self.HFTool(**params)
        return res