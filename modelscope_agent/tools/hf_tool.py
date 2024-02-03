from typing import Dict, List

from modelscope_agent.tools import register_tool
from modelscope_agent.tools.base import BaseTool
from transformers.tools import Tool as HFTool

from .tool import Tool


@register_tool('hf-tool')
class HFTool(BaseTool):
    """Simple wrapper for huggingface transformers tools

    """

    def __init__(self, tool: HFTool, description: str, name: str,
                 parameters: List[Dict]):
        self.tool = tool
        self.description = description
        self.name = name
        self.parameters = parameters
        super().__init__()

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        return json.dumps(self.tool(**params), ensure_ascii=False)
