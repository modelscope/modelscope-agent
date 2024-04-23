from modelscope_agent.tools.base import BaseTool


class MockTool(BaseTool):
    name: str = 'mock_tool'
    description: str = 'description'
    parameters: list = [{
        'name': 'test',
        'type': 'string',
        'description': 'test variable',
        'required': False
    }]

    def call(self, params: str, **kwargs):
        return params
