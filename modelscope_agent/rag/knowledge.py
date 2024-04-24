import os
from typing import Dict, Any, Union, List, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


@register_rag('base_pipeline')
class BaseKnowledgePipeline(BaseLlamaPack):
    """ base knowledge pipeline.

    从不同的源加载知识，支持：文件夹路径（str），文件路径列表（list），将不同源配置到不同的召回方式（dict）.
    Automatically select the best file reader given file extensions.

    Args:
        knowledges: Path to the directory，或文件路径列表，或指定召回方式的文件路径。
        save_path: 缓存indexing后的信息。
        llm: 总结召回内容时使用的llm。
    """

    def __init__(self,
                 knowledge_source: Union[Dict, List, str],
                 cache_dir: str = './run',
                 llm: Optional[LLM] = None,
                 **kwargs) -> None:
        extra_readers = self.get_extra_readers()
        documents = self.read(knowledge_source, extra_readers)

        if not documents:
            print('No valid document.')
            return

        # 为支持不同策略可自行挑选indexing的文档范围，indexing步骤也应在retrievers初始化内实现。
        retriever_tools = self.get_retriever_tools(documents, cache_dir)
        root_retriever = self.get_retriever_router(retriever_tools)
        self.query_engine = RetrieverQueryEngine.from_args(root_retriever)

    def get_transformations(self, **kwargs) -> Optional[List[TransformComponent]]:
        return None

    def get_retriever_tools(self, documents: List[Document], cache_dir: str) -> List[BaseRetriever]:
        retriever_tools = list()

        # indexing
        ## 可配置chunk_size等
        service_context = ServiceContext.from_defaults(chunk_size=256)
        ## 可对本召回器的文本范围 进行过滤、筛选、rechunk。transformations为空时，默认按语义rechunk。
        transformations = self.get_transformations()

        if cache_dir is not None and os.path.exists(cache_dir):
            # Load from cache
            from llama_index import StorageContext, load_index_from_storage
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
            # load index
            index = load_index_from_storage(storage_context)
        elif documents is not None:
            index = VectorStoreIndex.from_documents(documents=documents, transformations=transformations)
        else:
            index = VectorStoreIndex(documents=documents, transformations=transformations, service_context=service_context)

        if cache_dir is not None and not os.path.exists(cache_dir):
            index.storage_context.persist(persist_dir=cache_dir)

        # init retriever tool
        vector_retriever = index.as_retriever()

        ## 对召回后的内容进行处理
        retriever_postprocessors = self.get_postprocessors()

        ## 如果新增一个retriever，且使用同一个indexing（过滤、选择、rechunk策略都相同）
        # bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore)

        retriever_tools.append(
            RetrieverTool.from_defaults(
                retriever=vector_retriever,
                node_postprocessors=retriever_postprocessors))

        return retriever_tools

    def get_postprocessors(self, **kwargs) -> BaseNodePostprocessor:
        return None

    def get_retriever_router(self, retriever_tools: List[BaseRetriever]) -> RouterRetriever:
        router_retriever = RouterRetriever.from_default(retriever_tools)
        return router_retriever

    def get_extra_readers(self) -> Dict[str, BaseReader]:
        # lazy import
        try:
            from llama_index.readers.file import (
                PandasCSVReader,
                HTMLTagReader,
                FlatReader
            )
        except ImportError:
            raise ImportError("`llama-index-readers-file` package not found")

        return {'.pb': PandasCSVReader,
                '.html': HTMLTagReader,
                '.*': FlatReader}

    def read(self, knowledge_source: Union[str, List[str]], extra_readers: Dict[str, BaseReader]) -> List[Document]:
        if isinstance(knowledge_source):
            if os.path.isdir(knowledge_source):
                general_reader = SimpleDirectoryReader(input_dir=knowledge_source, file_extractor=extra_readers)
            elif os.path.isfile(knowledge_source):
                general_reader = SimpleDirectoryReader(input_files=[knowledge_source], file_extractor=extra_readers)
            else:
                raise ValueError(f'file path not exists: {knowledge_source}.')
        else:
            general_reader = SimpleDirectoryReader(input_files=knowledge_source, file_extractor=extra_readers)

        documents = general_reader.load_data(num_workers=os.cpu_count())
        return documents

    def run(self,
            query: str,
            **kwargs
            ) -> str:
        return self.query_engine.query(query, **kwargs)


