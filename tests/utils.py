from modelscope_agent.llm import LLM
from modelscope_agent.output_parser import OutputParser
from modelscope_agent.prompt import PromptGenerator
from modelscope_agent.tools import Tool


class MockLLM(LLM):

    def __init__(self, responses=['mock_llm_response']):
        super().__init__({})
        self.responses = responses
        self.idx = -1

    def generate(self, prompt: str) -> str:
        self.idx += 1
        return self.responses[self.idx] if self.idx < len(
            self.responses) else 'mock llm response'

    def stream_generate(self, prompt: str) -> str:
        yield 'mock llm response'


class MockPromptGenerator(PromptGenerator):

    def __init__(self):
        super().__init__()


class MockOutParser(OutputParser):

    def __init__(self, action, args):
        super().__init__()
        self.action = action
        self.args = args

    def parse_response(self, response: str):
        return self.action, self.args


class MockTool(Tool):

    def __init__(self, name, func, description, parameters=[]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
        super().__init__()

    def __call__(self, *args, **kwargs):

        return {'result': self.func(*args, **kwargs)}
