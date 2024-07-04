import json
from modelscope_agent.tools import BaseTool, register_tool

from .search_util import AuthenticationKey, get_websearcher_cls


@register_tool('web_search')
class WebSearch(BaseTool):
    description = 'surfacing relevant information from billions of web documents. Help ' \
                  'you find what you are looking for from the world-wide-web to comb ' \
                  'billions of webpages, images, videos, and news.'
    name = 'web_search'
    parameters: list = [{
        'name': 'query',
        'type': 'string',
        'description':
        """The user's search query term. The term may not be empty.""",
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        all_searchers = AuthenticationKey.to_dict()
        searcher = self.cfg.get('searcher', None)

        if not searcher:
            # set default as searcher
            from .searcher.bing import BingWebSearcher
            self.searcher = BingWebSearcher(**self.cfg)
        else:
            if isinstance(searcher, str) and len(searcher):
                cls = get_websearcher_cls(searcher)
                if not cls:
                    raise ValueError(
                        f'The searcher {searcher}\'s is not a valid name, please double check.'
                    )
                self.searcher = cls(**cfg)
            else:
                raise ValueError(
                    f'The searcher {searcher} should be one of {all_searchers.keys()}'
                )

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        res = self.searcher(query=params['query'], **kwargs)
        return json.dumps(res, ensure_ascii=False)


if __name__ == '__main__':
    tool = WebSearch()
    input_params = """{'query'='2024年 元旦 哈尔滨天气'}"""
    res = tool(input_params)
    print(res)
