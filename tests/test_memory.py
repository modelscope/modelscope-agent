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
    memory = MemoryWithRag(use_knowledge_cache=False, )

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
