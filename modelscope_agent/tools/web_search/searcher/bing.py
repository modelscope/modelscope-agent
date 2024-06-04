import json
import requests
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.web_search.search_util import SearchResult
from modelscope_agent.utils.utils import get_api_key

from .base_searcher import WebSearcher


class BingWebSearcher(WebSearcher):

    def __init__(
        self,
        timeout=3000,
        mkt='en-US',
        endpoint='https://api.bing.microsoft.com/v7.0/search',
        **kwargs,
    ):
        self.mkt = mkt
        self.endpoint = endpoint
        self.timeout = timeout

    def __call__(self, query, **kwargs):
        params = {'q': query, 'mkt': self.mkt}
        try:
            self.token = get_api_key(ApiNames.bing_api_key, **kwargs)
        except AssertionError:
            raise AssertionError(
                'bing web search api token must be acquired through ',
                'https://www.microsoft.com/en-us/bing/apis/bing-web-search-api',
                'and set by BING_SEARCH_V7_SUBSCRIPTION_KEY')

        headers = {'Ocp-Apim-Subscription-Key': self.token}
        if kwargs:
            params.update(kwargs)
        try:
            response = requests.get(
                self.endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout)
            raw_result = json.loads(response.text)
            if raw_result.get('error', None):
                print(f'Call Bing web search api failed: {raw_result}')
        except Exception as ex:
            raise ex('Call Bing web search api failed.')

        results = []
        res_list = raw_result.get('webPages', {}).get('value', [])
        for item in res_list:
            title = item.get('name', None)
            link = item.get('url', None)
            sniper = item.get('snippet', None)
            if not link and not sniper:
                continue

            results.append(
                SearchResult(title=title, link=link,
                             sniper=sniper).model_dump())

        return results


if __name__ == '__main__':
    searcher = BingWebSearcher()
    res = searcher('哈尔滨元旦的天气情况')
    print([item.__dict__ for item in res])
