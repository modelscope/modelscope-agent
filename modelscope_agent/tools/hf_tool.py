from typing import Dict, List

import json
from modelscope_agent.tools.base import BaseTool, register_tool


@register_tool('hf-tool')
class HFTool(BaseTool):
    """Simple wrapper for huggingface transformers tools

    """

    def __init__(self, tool, description: str, name: str,
                 parameters: List[Dict]):
        try:
            from transformers.tools import Tool as HFTool
        except ImportError:
            try:
                from transformers import Tool as HFTool
            except ImportError as e:
                raise ImportError(
                    "The package 'transformers' is required for this module. Please install it using 'pip install "
                    "transformers>=4.33'.") from e
        self.tool = tool
        self.description = description
        self.name = name
        self.parameters = parameters
        super().__init__()

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        return json.dumps(self.tool(**params), ensure_ascii=False)
