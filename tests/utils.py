from agent_scope.llm import BaseChatModel
from agent_scope.tools import Tool


class MockLLM(BaseChatModel):

    def __init__(self, responses=['mock_llm_response']):
        super().__init__({})
        self.responses = responses
        self.idx = -1
        self.model_id = 'mock_llm'

    def generate(self, prompt: str, function_list=[], **kwargs) -> str:
        self.idx += 1
        return self.responses[self.idx] if self.idx < len(
            self.responses) else 'mock llm response'

    def stream_generate(self, prompt: str, function_list=[], **kwargs) -> str:
        yield 'mock llm response'


class MockTool(Tool):

    def __init__(self, name, func, description, parameters=[]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
        super().__init__()

    def __call__(self, *args, **kwargs):

        return {'result': self.func(*args, **kwargs)}
