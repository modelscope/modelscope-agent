from modelscope_agent.memory import MemoryWithRag


def test_memory_with_rag_simple():

    memory = MemoryWithRag(
        urls=['tests/samples/常见QA.pdf'],
        use_knowledge_cache=False,
    )

    summary_str = memory.run(query='高德天气api怎么申请')
    print(summary_str)
    assert 'https://lbs.amap.com/api/javascript-api-v2/guide/services/weather' in summary_str


def test_memory_with_rag_update():
    memory = MemoryWithRag(use_knowledge_cache=False)

    summary_str = memory.run(
        query='模型大文件上传失败怎么办', url=['tests/samples/modelscope_qa_2.txt'])
    print(summary_str)
    assert 'git-lfs' in summary_str


def test_memory_with_rag_multi_sources():

    memory = MemoryWithRag(
        urls=['tests/samples/modelscope_qa_2.txt', 'tests/samples/常见QA.pdf'],
        use_knowledge_cache=False,
    )

    summary_str = memory.run('高德天气api怎么申请')
    print(summary_str)
    assert 'https://lbs.amap.com/api/javascript-api-v2/guide/services/weather' in summary_str


def test_memory_with_rag_cache():
    MemoryWithRag(
        urls=['tests/samples/modelscope_qa_2.txt', 'tests/samples/常见QA.pdf'],
        storage_path='./tmp/',
        use_knowledge_cache=False,
    )

    memory = MemoryWithRag(
        storage_path='./tmp/',
        use_knowledge_cache=True,
    )
    summary_str = memory.run('模型大文件上传失败怎么办')
    print(summary_str)
    assert 'git-lfs' in summary_str


def test_memory_with_rag_custom():
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.readers.json import JSONReader
    from llama_index.core.extractors import TitleExtractor

    memory = MemoryWithRag(
        urls=['tests/samples/modelscope_qa_2.txt'],
        storage_path='./tmp/',
        memory_path='./tmp/',
        use_knowledge_cache=False,
        retriever=BM25Retriever,
        loaders={'.json': JSONReader},
        post_processors=[],
        transformations=[TitleExtractor])
    summary_str = memory.run('模型大文件上传失败怎么办')
    print(summary_str)
    assert 'git-lfs' in summary_str


def test_memory_with_rag_multi_modal():
    memory = MemoryWithRag(
        urls=['tests/samples/rag.png'],
        use_knowledge_cache=False,
    )

    summary_str = memory.run('我想看rag的流程图')
    print(summary_str)
    assert 'rag.png' in summary_str


def test_memory_with_rag_no_use_llm():
    memory = MemoryWithRag(use_knowledge_cache=False)

    summary_str = memory.run(
        query='模型大文件上传失败怎么办',
        url=['tests/samples/modelscope_qa_2.txt'],
        use_llm=False)
    print(summary_str)
    assert 'file_path' in summary_str
    assert 'git-lfs' in summary_str


def test_memory_with_rag_mongodb_storage():
    # $ mongo
    import os
    import shutil
    from llama_index.storage.docstore.mongodb import MongoDocumentStore
    from llama_index.storage.index_store.mongodb import MongoIndexStore
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost')
    cache_dir = './tmp'

    MemoryWithRag(
        urls=['tests/samples/modelscope_qa_1.txt'],
        storage_path=cache_dir,
        use_knowledge_cache=True,
        docstore=MongoDocumentStore.from_uri(MONGO_URI),
        index_store=MongoIndexStore.from_uri(MONGO_URI))

    # clean local cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    memory = MemoryWithRag(
        use_knowledge_cache=True,
        storage_path=cache_dir,
        docstore=MongoDocumentStore.from_uri(MONGO_URI),
        index_store=MongoIndexStore.from_uri(MONGO_URI))
    summary_str = memory.run('环境安装报错 missing xcrun 怎么办？')
    print(summary_str)
    assert 'xcode-select --install' in summary_str


def test_memory_with_rag_mongodb_storage():
    from pathlib import Path
    from llama_index.readers.file import FlatReader
    reader = FlatReader()
    documents = reader.load_data(Path('tests/samples/code.py'))
    memory = MemoryWithRag(use_knowledge_cache=False, documents=documents)

    res = memory.run('BaseRetriever的import路径是哪?', use_llm=False)
    print(res)
    assert 'llama_index.core.base.base_retriever' in res
