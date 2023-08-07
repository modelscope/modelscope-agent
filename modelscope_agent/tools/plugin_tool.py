from copy import deepcopy

from langchain.tools import BaseTool

from .tool import Tool


class LangchainTool(Tool):

    def __init__(self, langchain_tool):
        if not isinstance(langchain_tool, BaseTool):
            raise ValueError('langchain_tool should be type of langchain tool')
        self.langchain_tool = langchain_tool
        self.parse_langchain_schema()
        super().__init__()

    def parse_langchain_schema(self):
        # convert langchain tool schema to modelscope_agent tool schema
        self.description = self.langchain_tool.description
        self.name = self.langchain_tool.name
        self.parameters = []
        for name, arg in self.langchain_tool.args.items():
            tool_arg = deepcopy(arg)
            tool_arg['name'] = name
            tool_arg['required'] = True
            tool_arg.pop('title')
            self.parameters.append(tool_arg)

    def _local_call(self, *args, **kwargs):
        return {'result': self.langchain_tool.run(kwargs)}
