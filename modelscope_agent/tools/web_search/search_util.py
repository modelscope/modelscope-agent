import os
from enum import Enum

from modelscope_agent.constants import ApiNames
from pydantic import BaseModel, model_validator


class SearchResult(BaseModel):
    title: str
    link: str = None
    sniper: str = None

    @model_validator(mode='before')
    def validate_values(self):
        if self['link'] is None and self['sniper'] is None:
            raise ValueError('Either link or sniper must be provided.')
        return self


class AuthenticationKey(Enum):
    bing = ApiNames.bing_api_key.value
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
        from .searcher.bing import BingWebSearcher
        cls_dict[AuthenticationKey.bing.name] = BingWebSearcher
    if get_env(AuthenticationKey.kuake.value):
        from .searcher.kuake import KuakeWebSearcher
        cls_dict[AuthenticationKey.kuake.name] = KuakeWebSearcher

    return cls_dict
