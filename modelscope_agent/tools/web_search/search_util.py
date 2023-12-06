import os


class SearchResult:

    def __init__(self, title=None, link=None, sniper=None):
        assert link or sniper
        self.title = title
        self.link = link
        self.sniper = sniper


class AuthenticationKey:
    bing = 'BING_SEARCH_V7_SUBSCRIPTION_KEY'
    kuake = 'PLACE_HOLDER'

    @classmethod
    def to_dict(cls):
        raw_dict = cls.__dict__
        res = dict(
            filter(lambda x: '__' not in x[0] and isinstance(x[1], str),
                   raw_dict.items()))
        return res


def get_websearcher_cls():

    def get_env(authentication_key: str):
        env = os.environ
        return env.get(authentication_key, None)

    cls_list = []
    if get_env(AuthenticationKey.bing):
        from modelscope_agent.tools.web_search.searcher.bing import BingWebSearcher
        cls_list.append(BingWebSearcher)
    if get_env(AuthenticationKey.kuake):
        from modelscope_agent.tools.web_search.searcher.kuake import KuakeWebSearcher
        cls_list.append(KuakeWebSearcher)

    return cls_list
