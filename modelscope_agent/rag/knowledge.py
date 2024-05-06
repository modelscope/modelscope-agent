import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers.base import BaseReader
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.schema import Document, QueryBundle, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.tools.retriever_tool import RetrieverTool
from llama_index.core.vector_stores.types import (MetadataFilter,
                                                  MetadataFilters)
from modelscope_agent.llm.dashscope import DashScopeLLM
from modelscope_agent.rag.emb.dashscope import DashscopeEmbedding
from modelscope_agent.rag.llm import MSAgentLLM

#from modelscope_agent.rag.selector import FileSelector


@dataclass
class FileQueryBundle(QueryBundle):
    files: List[str] = None


# @register_rag('base_knowledge')
class BaseKnowledge(BaseLlamaPack):
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
                 llm: Optional[DashScopeLLM] = None,
                 **kwargs) -> None:
        extra_readers = self.get_extra_readers()
        documents = self.read(knowledge_source, extra_readers)

        if not documents:
            print('No valid document.')
            return

        if llm:
            llm = MSAgentLLM(llm)

        # 为支持不同策略可自行挑选indexing的文档范围，indexing步骤也应在retrievers初始化内实现。
        root_retriever = self.get_root_retriever(documents, cache_dir, llm=llm)
        self.query_engine = RetrieverQueryEngine.from_args(
            root_retriever, llm=llm)

    def get_transformations(self,
                            **kwargs) -> Optional[List[TransformComponent]]:
        # rechunk，筛选文档内容等
        return None

    def _get_retriever_tools(self, documents: List[Document],
                             cache_dir: str) -> List[BaseRetriever]:
        retriever_tools = list()

        # indexing
        ## 可配置chunk_size等
        Settings.chunk_size = 512
        ## 可对本召回器的文本范围 进行过滤、筛选、rechunk。transformations为空时，默认按语义rechunk。
        transformations = self.get_transformations()
        if cache_dir is not None and os.path.exists(cache_dir):
            # Load from cache
            from llama_index.core import StorageContext, load_index_from_storage
            # rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=cache_dir)
            # load index
            index = load_index_from_storage(
                storage_context, embed_model=DashscopeEmbedding())
        elif documents is not None:
            index = VectorStoreIndex.from_documents(
                documents=documents,
                transformations=transformations,
                embed_model=DashscopeEmbedding())
        else:
            print('Neither documents nor cache_dir.')
            # index = VectorStoreIndex(nodes, transformations=transformations, embed_model=DashscopeEmbedding())

        if cache_dir is not None and not os.path.exists(cache_dir):
            index.storage_context.persist(persist_dir=cache_dir)

        # init retriever tool
        vector_retriever = index.as_retriever()

        return vector_retriever

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
        # 获取召回内容后处理器
        return None

    def get_root_retriever(self, documents: List[Document], cache_dir: str,
                           llm: LLM) -> BaseRetriever:
        # retriever_tools = self._get_retriever_tools(documents, cache_dir)
        #selector = self.get_retriever_selector()
        #router_retriever = RouterRetriever(retriever_tools, llm=llm, selector=selector)
        return self._get_retriever_tools(documents, cache_dir)

    def get_retriever_selector(self, **kwargs) -> BaseSelector:
        # 根据query选择使用哪些retriever
        return None

    def get_extra_readers(self) -> Dict[str, BaseReader]:
        # lazy import
        try:
            from llama_index.readers.file import (PandasCSVReader,
                                                  HTMLTagReader, FlatReader)
        except ImportError:
            print(
                '`llama-index-readers-file` package not found. Can not read .pd .html file.'
            )
            return {}

        return {'.pb': PandasCSVReader, '.html': HTMLTagReader}

    def read(self, knowledge_source: Union[str, List[str]],
             extra_readers: Dict[str, BaseReader]) -> List[Document]:
        if isinstance(knowledge_source, str):
            if os.path.isdir(knowledge_source):
                general_reader = SimpleDirectoryReader(
                    input_dir=knowledge_source, file_extractor=extra_readers)
            elif os.path.isfile(knowledge_source):
                general_reader = SimpleDirectoryReader(
                    input_files=[knowledge_source],
                    file_extractor=extra_readers)
            else:
                raise ValueError(f'file path not exists: {knowledge_source}.')
        else:
            general_reader = SimpleDirectoryReader(
                input_files=knowledge_source, file_extractor=extra_readers)

        documents = general_reader.load_data(num_workers=os.cpu_count())
        return documents

    def set_filter(self, files: List[str]):
        retriever = self.query_engine.retriever
        filters = [
            MetadataFilter(key='file_name', value=file) for file in files
        ]
        retriever._filters = MetadataFilters(filters=filters)
        print(retriever._filters)

    def run(self, query: str, files: List[str]=[], **kwargs) -> str:
        query_bundle = FileQueryBundle(query)

        if len(files) > 0:
            self.set_filter(files)

        return self.query_engine.query(query_bundle, **kwargs)


if __name__ == '__main__':
    from modelscope_agent.llm import get_chat_model
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
    llm = get_chat_model(**llm_config)
    knowledge_source = ['data/Agent.pdf', 'data/QA.pdf']

    knowledge = BaseKnowledge('./data', llm=llm)
    
    print(knowledge.run('高德天气API申请', files=['QA.pdf']))
    print('-----------------------')
    print(knowledge.run('高德天气API申请', files=['Agent.pdf']))
