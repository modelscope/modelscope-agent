from modelscope_agent.tools.web_search.base_searcher import WebSearcher


class KuakeWebSearcher(WebSearcher):

    def __call__(self, query, **kwargs):
        raise NotImplementedError()
