from modelscope_agent.action_parser import ActionParser
from modelscope_agent.llm import LLM
from modelscope_agent.prompt import PromptGenerator
from modelscope_agent.tools import Tool


class MockLLM(LLM):

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


class MockPromptGenerator(PromptGenerator):

    def __init__(self):
        super().__init__()


class MockOutParser(ActionParser):

    def __init__(self, action, args, count=1):
        super().__init__()
        self.action = action
        self.args = args
        self.count = count

    def parse_response(self, response: str):
        if self.count > 0:
            self.count -= 1
            return self.action, self.args
        else:
            return None, None


class MockTool(Tool):

    def __init__(self, name, func, description, parameters=[]):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters
        super().__init__()

    def __call__(self, *args, **kwargs):

        return {'result': self.func(*args, **kwargs)}
