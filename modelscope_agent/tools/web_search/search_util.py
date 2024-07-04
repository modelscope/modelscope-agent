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


def get_websearcher_cls(searcher: str = None):

    if searcher:
        if AuthenticationKey.bing.name == searcher:
            from .searcher.bing import BingWebSearcher
            return BingWebSearcher
        elif AuthenticationKey.kuake.name == searcher:
            from .searcher.kuake import KuakeWebSearcher
            return KuakeWebSearcher
    return None
