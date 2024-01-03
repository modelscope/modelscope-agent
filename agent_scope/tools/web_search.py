import os

from agent_scope.tools.tool import Tool, ToolSchema
from agent_scope.tools.web_search_utils import get_websearcher_cls
from agent_scope.tools.web_search_utils.search_util import AuthenticationKey
from pydantic import ValidationError


class WebSearch(Tool):
    description = 'surfacing relevant information from billions of web documents. Help ' \
                  'you find what you are looking for from the world-wide-web to comb ' \
                  'billions of webpages, images, videos, and news.'
    name = 'web_search'
    parameters: list = [{
        'name': 'query',
        'description':
        """The user's search query term. The term may not be empty.""",
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        available_searchers = get_websearcher_cls()
        all_searchers = AuthenticationKey.to_dict()
        if not len(available_searchers):
            raise ValueError(
                f'At least one of web search api token should be set: {all_searchers}'
            )

        searcher = self.cfg.get('searcher', None)

        if not searcher:
            self.searcher = available_searchers[0](**self.cfg)
        else:
            if isinstance(searcher,
                          str) and len(searcher) and all_searchers.get(
                              searcher, None):
                cls = available_searchers.get(searcher, None)
                if not cls:
                    raise ValueError(
                        f'The searcher {searcher}\'s token is not set: {all_searchers.get(searcher, None)}'
                    )
                self.searcher = cls(**cfg)
            else:
                raise ValueError(
                    f'The searcher {searcher} should be one of {all_searchers.keys()}'
                )

        try:
            all_para = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_para)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self.is_remote_tool = True
        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(all_para)

    def _remote_call(self, *args, **kwargs):
        query = self._handle_input_fallback(**kwargs)
        if not query or not len(query):
            raise ValueError(
                'parameter `query` of tool web-search is None or Empty.')

        res = self.searcher(query)
        return {'result': [item.__dict__ for item in res]}

    def _handle_input_fallback(self, **kwargs):
        query = kwargs.get('query', None)
        fallback = kwargs.get('fallback', None)
        if query and isinstance(query, str) and len(query):
            return query
        else:
            return fallback


if __name__ == '__main__':
    tool = WebSearch()
    res = tool(query='2024年 元旦 哈尔滨天气')
    print(res)
