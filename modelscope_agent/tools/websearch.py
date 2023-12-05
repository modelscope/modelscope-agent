import os

from modelscope_agent.tools.web_search.search_util import AuthenticationKey
from modelscope_agent.tools.web_search import get_websearcher_cls
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError


class WebSearch(Tool):
    description = 'surfacing relevant information from billions of web documents. Help you find what you are looking for from the world-wide-web to comb billions of webpages, images, videos, and news.'
    name = 'web_search'
    parameters: list = [{
        'name': 'query',
        'description': """The user's search query term. The term may not be empty.""",
        'required': True
    }]

    def __init__(self, cfg={}):
        available_searchers = get_websearcher_cls()
        if not len(available_searchers):
            raise ValueError(f'At least one of web search api token should be set: {AuthenticationKey.__dict__}')
        self.searcher = available_searchers[0]()

        try:
            all_para = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_para)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(all_para)

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query', None)
        if not query or not len(query):
            raise ValueError(f'parameter `query` of tool web-search is None or Empty.')

        res = self.searcher(query)
        return {'result': [item.__dict__ for item in res]}

    def _handle_input_fallback(self, param_input: str):
        return self.__call__(query=param_input)


if __name__ == '__main__':
    tool = WebSearch()
    res = tool(query='2024年 元旦 哈尔滨天气')
    print(res)
