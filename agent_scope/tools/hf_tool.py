from typing import Dict, List

from transformers.tools import Tool as HFTool

from .tool import Tool


class HFTool(Tool):
    """Simple wrapper for huggingface transformers tools

    """

    def __init__(self, tool: HFTool, description: str, name: str,
                 parameters: List[Dict]):
        self.tool = tool
        self.description = description
        self.name = name
        self.parameters = parameters
        super().__init__()

    def _local_call(self, *args, **kwargs):
        return {'result': self.tool(**kwargs)}
