import httpx
from langchain.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modelscope_agent.tools.tool import Tool


class WebBrowser(Tool):
    description = '调用web browser api处理网页内容'
    name = 'web_browser'
    parameters: list = [{
        'name': 'urls',
        'description': 'the urls that the user wants to browse',
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        self.split_url_into_chunk = self.cfg.get('split_url_into_chunk', False)
        self.max_browser_length = self.cfg.get('max_browser_length', 2000)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
        self.client = httpx.Client(
            headers=self.headers, verify=False, timeout=30.0)

    def _local_call(self, *args, **kwargs):
        parsed_args, parsed_kwargs = self._local_parse_input(*args, **kwargs)

        urls = parsed_kwargs['urls']
        print(urls)
        if urls is None:
            return {'result': ''}

        # # load html
        loader = AsyncChromiumLoader(urls)
        docs = loader.load()
        # Transform
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs, tags_to_extract=['span', 'div', 'p'])

        # split url content into chunk in order to get fine-grained results
        if self.split_url_into_chunk:
            # Grab the first 1000 tokens of the site
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=0)
            splits = splitter.split_documents(docs_transformed)
        else:
            splits = docs_transformed
        search_results = []
        for item in splits:
            result = {
                'url': item.metadata['source'],
                'content': item.page_content[
                    0:self.max_browser_length]  # make it maximum 2000k
            }
            search_results.append(result)

        return {'result': search_results}

    def _local_parse_input(self, *args, **kwargs):
        urls = kwargs.get('urls', [])
        if isinstance(urls, str):
            urls = [urls]
        kwargs['urls'] = urls
        return args, kwargs


if __name__ == '__main__':
    tool = WebBrowser()
    urls = ['https://blog.sina.com.cn/zhangwuchang']
    result = tool._local_call(urls=urls)
    print(result)
