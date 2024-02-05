import re

from modelscope_agent.utils.nltk_utils import install_nltk_data
from modelscope_agent.utils.tokenization_utils import count_tokens


def rm_newlines(text):
    text = re.sub(r'(?<=[^\.。:：])\n', ' ', text)
    return text


def rm_cid(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    return text


def rm_hexadecimal(text):
    text = re.sub(r'[0-9A-Fa-f]{21,}', '', text)
    return text


def deal(text):
    text = rm_newlines(text)
    text = rm_cid(text)
    text = rm_hexadecimal(text)
    return text


def parse_doc(path):
    try:
        if '.pdf' in path.lower():
            from langchain_community.document_loaders import PDFMinerLoader
            loader = PDFMinerLoader(path)
            pages = loader.load_and_split()
        elif '.docx' in path.lower():
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(path)
            pages = loader.load_and_split()
        elif '.pptx' in path.lower():
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(path)
            pages = loader.load_and_split()
        elif '.txt' in path.lower():
            from langchain_community.document_loaders import TextLoader
            from langchain.text_splitter import CharacterTextSplitter
            # split the knowledge into chunk size 1000
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            loader = TextLoader(path, autodetect_encoding=True)
            pages = loader.load_and_split(text_splitter)
        else:
            from langchain_community.document_loaders import UnstructuredFileLoader
            install_nltk_data()
            loader = UnstructuredFileLoader(path)
            pages = loader.load_and_split()
    except Exception as e:
        import traceback
        raise RuntimeError(
            f'Failed to load document with error {e}, and detail: {traceback.format_exc()}'
        )

    res = []
    for page in pages:
        dealed_page_content = deal(page.page_content)
        res.append({
            'page_content': dealed_page_content,
            'token': count_tokens(dealed_page_content),
            'metadata': page.metadata
        })

    return res


def pre_process_html(s):
    # replace multiple newlines
    s = re.sub('\n+', '\n', s)
    # replace special string
    s = s.replace("Add to Qwen's Reading List", '')
    return s


def parse_html_bs(path):
    from langchain_community.document_loaders import BSHTMLLoader

    loader = BSHTMLLoader(path, open_encoding='utf-8')
    pages = loader.load_and_split()
    res = []
    for page in pages:
        dealed_page_content = pre_process_html(page.page_content)
        res.append({
            'page_content': dealed_page_content,
            'token': count_tokens(dealed_page_content),
            'metadata': page.metadata
        })

    return res
