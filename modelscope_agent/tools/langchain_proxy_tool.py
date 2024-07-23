from copy import deepcopy

from modelscope_agent.tools.base import BaseTool, register_tool


@register_tool('langchain_tool')
class LangchainTool(BaseTool):
    description = '通过调用langchain插件来支持对语言模型的输入输出格式进行处理，输入文本字符，输出经过格式处理的结果'
    name = 'plugin'
    parameters: list = [{
        'name': 'commands',
        'description': '需要进行格式处理的文本字符列表',
        'required': True,
        'type': 'string'
    }]

    def __init__(self, langchain_tool):
        from langchain_community.tools import BaseTool

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
            if 'type' not in arg:
                tool_arg['type'] = arg['anyOf'][0].get('type', 'string')
            tool_arg.pop('title')
            self.parameters.append(tool_arg)

    def call(self, params: str, **kwargs):
        params = self._verify_args(params)
        res = self.langchain_tool.run(params)
        return res
