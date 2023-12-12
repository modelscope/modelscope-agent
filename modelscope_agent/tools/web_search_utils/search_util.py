import os
from enum import Enum


class SearchResult:

    def __init__(self, title=None, link=None, sniper=None):
        assert link or sniper
        self.title = title
        self.link = link
        self.sniper = sniper


class AuthenticationKey(Enum):
    bing = 'BING_SEARCH_V7_SUBSCRIPTION_KEY'
    kuake = 'PLACE_HOLDER'

    @classmethod
    def to_dict(cls):
        return {member.name: member.value for member in cls}


def get_websearcher_cls():

    def get_env(authentication_key: str):
        env = os.environ
        return env.get(authentication_key, None)

    cls_dict = {}
    if get_env(AuthenticationKey.bing.value):
        from modelscope_agent.tools.web_search_utils.searcher.bing import BingWebSearcher
        cls_dict[AuthenticationKey.bing.name] = BingWebSearcher
    if get_env(AuthenticationKey.kuake.value):
        from modelscope_agent.tools.web_search_utils.searcher.kuake import KuakeWebSearcher
        cls_dict[AuthenticationKey.kuake.name] = KuakeWebSearcher

    return cls_dict
