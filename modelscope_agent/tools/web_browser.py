import base64

import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (AsyncChromiumLoader,
                                                  AsyncHtmlLoader)
from langchain_community.document_transformers import BeautifulSoupTransformer
from modelscope_agent.tools.base import BaseTool, register_tool


@register_tool('web_browser')
class WebBrowser(BaseTool):
    description = '网页浏览器，能根据网页url浏览网页并返回网页内容。'
    name = 'web_browser'
    parameters: list = [{
        'name': 'urls',
        'type': 'string',
        'description': 'the urls that the user wants to browse',
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
        self.client = httpx.Client(
            headers=self.headers, verify=False, timeout=30.0)
        self.cfg = cfg.get(self.name, {})

        self.use_advantage = self.cfg.get('use_adv', False)

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        urls = params['urls']
        print(urls)
        if urls is None:
            return ''

        if self.use_advantage:
            text_result, image_result = self.advantage_https_get(
                urls, **kwargs)
        else:
            text_result = self.simple_https_get(urls, **kwargs)
        return text_result

    def advantage_https_get(self, urls, **kwargs):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return (
                'Please install playwright with chromium kernel by running `pip install playwright` and '
                '`playwright install --with-deps chromium`')

        if isinstance(urls, list):
            urls = urls[0]
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(urls)
            text_result = page.evaluate('() => document.body.innerText')
            screenshot_bytes = page.screenshot(full_page=True)
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode(
                'utf-8')
            browser.close()

        return text_result, screenshot_base64

    def simple_https_get(self, urls, **kwargs):
        # load html and get docs
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        result = self._post_process(docs, **kwargs)
        return result

    def _post_process(self, docs, **kwargs):
        # make sure parameters could be initialized in runtime
        max_browser_length = kwargs.get('max_browser_length', 2000)
        split_url_into_chunk = kwargs.get('split_url_into_chunk', False)

        # Transform
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs, tags_to_extract=['span'])

        # split url content into chunk in order to get fine-grained results
        if split_url_into_chunk:
            # Grab the first 1000 tokens of the site
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=0)
            splits = splitter.split_documents(docs_transformed)
        else:
            splits = docs_transformed
        search_results = ''
        for item in splits:
            search_results += item.page_content + '\n'

        return search_results[0:max_browser_length]


if __name__ == '__main__':
    tool = WebBrowser()
    result = tool.call('{"urls": "https://blog.sina.com.cn/zhangwuchang"}')
    print(result)
