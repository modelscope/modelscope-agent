import os

from modelscope_agent.tools.web_search.search_util import (AuthenticationKey,
                                                           SearchResult)


class WebSearcher:
    timeout = 1000

    def __call__(self, args):
        raise NotImplementedError()
