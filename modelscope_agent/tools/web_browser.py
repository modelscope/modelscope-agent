import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (AsyncChromiumLoader,
                                                  AsyncHtmlLoader)
from langchain_community.document_transformers import BeautifulSoupTransformer
from modelscope_agent.tools import BaseTool, register_tool


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

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        urls = params['urls']
        print(urls)
        if urls is None:
            return ''

        # make sure parameters could be initialized in runtime
        max_browser_length = kwargs.get('max_browser_length', 2000)
        split_url_into_chunk = kwargs.get('split_url_into_chunk', False)

        # # load html
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
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
