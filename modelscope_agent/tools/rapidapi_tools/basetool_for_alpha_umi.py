from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import json
import json5
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.utils.utils import has_chinese_chars

TOOL_REGISTRY = {}


def register_tool(name):

    def decorator(cls):
        TOOL_REGISTRY[name] = cls
        return cls

    return decorator


class BasetoolAlphaUmi(BaseTool):
    name: str
    description: str
    parameters: List[Dict]

    def __init__(self, cfg: Optional[Dict] = {}):
        """
        :param schema: Format of tools, default to oai format, in case there is a need for other formats
        """
        self.cfg = cfg.get(self.name, {})

        self.schema = 'alpha_umi'
        self.function = self._build_function()
        self.function_plain_text = self._parser_function()

    def _build_function(self):
        """
        The dict format after applying the template to the function, such as oai format

        """
        input_doc = {}
        for p in self.parameters:
            input_doc[p['name']] = (
                p['type'] + ', '
                + 'required, ' if p['required'] else 'optional, '
                + p['description'][:128])

        function = {
            'Name': self.name[-64:],
            'function': self.description[-256:],
            'input': input_doc
        }

        return function

    def _parser_function(self):
        """
        Text description of function

        """

        return json.dumps(self.function)
