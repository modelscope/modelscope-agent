import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Type

import fsspec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, QueryBundle, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (MetadataFilter,
                                                  MetadataFilters)
from modelscope_agent.llm import get_chat_model
from modelscope_agent.llm.dashscope import DashScopeLLM
from modelscope_agent.rag.emb.dashscope import DashscopeEmbedding
from modelscope_agent.rag.llm import MSAgentLLM


@dataclass
class FileQueryBundle(QueryBundle):
    files: List[str] = None


# @register_rag('base_knowledge')
class BaseKnowledge(BaseLlamaPack):
    """ base knowledge pipeline.

    从不同的源加载知识，支持：文件夹路径（str），文件路径列表（list），将不同源配置到不同的召回方式（dict）.
    Automatically select the best file reader given file extensions.

    Args:
        knowledge_source: Path to the directory，或文件路径列表，或指定召回方式的文件路径。
        cache_dir: 缓存indexing后的信息。
        llm: 总结召回内容时使用的llm。
    """

    def __init__(self,
                 knowledge_source: Union[Dict, List, str],
                 cache_dir: str = './run',
                 llm: Optional[DashScopeLLM] = None,
                 retriever: Optional[Type[BaseRetriever]] = None,
                 loaders: Dict[str, Type[BaseReader]] = {},
                 transformations: List[Type[TransformComponent]] = [],
                 post_processors: List[Type[BaseNodePostprocessor]] = [],
                 **kwargs) -> None:
        extra_readers = self.get_extra_readers(loaders)
        documents = self.read(knowledge_source, extra_readers)

        if not documents:
            print('No valid document.')
            return

        if llm and isinstance(llm, DashScopeLLM):
            llm = MSAgentLLM(llm)
        elif isinstance(llm, LLM):
            pass
        else:
            llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
            llm = get_chat_model(**llm_config)
            llm = MSAgentLLM(llm)

        # 可对本召回器的文本范围 进行过滤、筛选、rechunk。transformations为空时，默认按语义rechunk。
        transformations = self.get_transformations(transformations)
        root_retriever = self.get_root_retriever(
            documents,
            cache_dir,
            transformations=transformations,
            llm=llm,
            retriever=retriever,
            **kwargs)
        postprocessors = self.get_postprocessors(post_processors, **kwargs)
        self.query_engine = self.get_query_engine(root_retriever, llm,
                                                  postprocessors, **kwargs)

    def get_query_engine(self, root_retriever: BaseRetriever, llm: LLM,
                         postprocessors, **kwargs) -> BaseQueryEngine:
        return RetrieverQueryEngine.from_args(
            root_retriever, llm=llm, node_postprocessors=postprocessors)

    def get_transformations(self, transformations: List[Type[TransformComponent]],
                            **kwargs) -> Optional[List[TransformComponent]]:
        # rechunk，筛选文档内容等
        res = []
        for t in transformations:
            try:
                t.from_defaults()
                res.append(res)
            except Exception as e:
                print(f'node parser {t} cannot be used and it will be ignored. Detail: {e}')
        return res

    def get_postprocessors(self, post_processors: List[Type[BaseNodePostprocessor]],
                           **kwargs) -> Optional[List[Type[BaseNodePostprocessor]]]:
        # 获取召回内容后处理器
        res = []
        for post_processor_cls in post_processors:
            try:
                post_processor = post_processor_cls()
                res.append(post_processor)
            except Exception as e:
                print(f'post_processor_cls {post_processor_cls} cannot be used and it will be ignored. Detail: {e}')

        return res

    def get_root_retriever(self, documents: List[Document], cache_dir: str,
                           transformations: Optional[List[TransformComponent]],
                           llm: LLM, retriever: Optional[Type[BaseRetriever]] = None, **kwargs) -> BaseRetriever:

        # indexing
        # 可配置chunk_size等
        Settings.chunk_size = 512
        index = None
        if cache_dir is not None and os.path.exists(cache_dir):
            try:
                # Load from cache
                from llama_index.core import StorageContext, load_index_from_storage
                # rebuild storage context
                storage_context = StorageContext.from_defaults(
                    persist_dir=cache_dir)
                # load index

                index = load_index_from_storage(
                    storage_context, embed_model=DashscopeEmbedding())
            except Exception as e:
                print(
                    f'Can not load index from cache_dir {cache_dir}, detail: {e}'
                )
        if documents is not None:
            if not index:
                index = VectorStoreIndex.from_documents(
                    documents=documents,
                    transformations=transformations,
                    embed_model=DashscopeEmbedding())
            else:
                for doc in documents:
                    index.insert(doc)
        if not index:
            print('Neither documents nor cache_dir.')
            # index = VectorStoreIndex(nodes, transformations=transformations, embed_model=DashscopeEmbedding())

        if cache_dir is not None:
            index.storage_context.persist(persist_dir=cache_dir)

        # init retriever tool
        if retriever:
            try:
                return retriever.from_defaults(index)
            except Exception as e:
                print(f'Retriever {retriever} cannot be used, using default retriever instead. Detail: {e}')

        return index.as_retriever()

    def get_extra_readers(self, loaders: Dict[str, Type[BaseReader]]) -> Dict[str, BaseReader]:
        extra_readers = {}
        for file_type, loader_cls in loaders.items():
            try:
                loader = loader_cls()
                extra_readers[file_type] = loader
            except Exception as e:
                print(f'Using {loader_cls} failed. Can not read {file_type} file. Detail: {e}')
        
        # lazy import
        try:
            from llama_index.readers.file import (PandasCSVReader,
                                                  HTMLTagReader, FlatReader)
        except ImportError:
            print(
                '`llama-index-readers-file` package not found. Can not read .pd .html .txt file.'
            )
            return extra_readers

        return {
            '.pb': PandasCSVReader(),
            '.html': HTMLTagReader(),
            '.txt': FlatReader()
        }.update(extra_readers)

    def read(
        self,
        knowledge_source: Union[str,
                                List[str]],  # file_dir or list of file_path
        extra_readers: Dict[
            str,
            BaseReader],  # extra_readers get from self.get_extra_readers()
        exclude_hidden:
        bool = True,  # Whether to exclude hidden files (dotfiles).
        recursive:
        bool = False,  # Whether to recursively search in subdirectories.
        fs: Optional[
            fsspec.
            AbstractFileSystem] = None,  # File system to use. Defaults to using the local file system. Can be changed to use any remote file system exposed via the fsspec interface.
        **kwargs
    ) -> List[Document]:

        try:
            if isinstance(knowledge_source, str):
                if os.path.isdir(knowledge_source):
                    general_reader = SimpleDirectoryReader(
                        input_dir=knowledge_source,
                        file_extractor=extra_readers,
                        exclude_hidden=exclude_hidden,
                        fs=fs,
                        recursive=recursive)
                elif os.path.isfile(knowledge_source):
                    general_reader = SimpleDirectoryReader(
                        input_files=[knowledge_source],
                        file_extractor=extra_readers,
                        exclude_hidden=exclude_hidden,
                        fs=fs,
                        recursive=recursive)
                else:
                    raise ValueError(
                        f'file path not exists: {knowledge_source}.')
            else:
                general_reader = SimpleDirectoryReader(
                    input_files=knowledge_source,
                    file_extractor=extra_readers,
                    fs=fs,
                    exclude_hidden=exclude_hidden,
                    recursive=recursive)

            documents = general_reader.load_data(num_workers=os.cpu_count())
        except ValueError:
            print('No valid documents')
            documents = []
        return documents

    def set_filter(self, files: List[str]):
        retriever = self.query_engine.retriever
        filters = [
            MetadataFilter(key='file_name', value=file) for file in files
        ]
        retriever._filters = MetadataFilters(filters=filters)
        print(retriever._filters)

    def run(self, query: str, files: List[str] = [], **kwargs) -> str:
        query_bundle = FileQueryBundle(query)

        if len(files) > 0:
            self.set_filter(files)

        return str(self.query_engine.query(query_bundle, **kwargs))

    def add_files(self, files: List[str]):

        if isinstance(files, str):
            files = [files]

        try:
            extra_readers = self.get_extra_readers()
            docs = self.read(files, extra_readers)
            for doc in docs:
                self.query_engine.retriever._index.insert(doc)

        except BaseException as e:
            print(f'add files {files} failed, detail: {e}')

    def delete_file(self, file: str):
        pass


if __name__ == '__main__':
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
    llm = get_chat_model(**llm_config)

    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.readers.json import JSONReader
    #from llama_index.postprocessor.cohere_rerank import CohereRerank
    from llama_index.legacy.node_parser.file.markdown import MarkdownNodeParser
    knowledge = BaseKnowledge('./data', llm=llm, retriever=BM25Retriever, loaders={'.json': JSONReader}, post_processors=[], transformations=[MarkdownNodeParser])

    import time
    s = time.time()
    print("start time", s)
    print(knowledge.run('高德天气API申请'))
    e = time.time()
    print("end time", e)
    print('-----------------------')

    knowledge.add_files('./data2/常见QA.pdf')
    print(knowledge.run('高德天气API申请', files=['常见QA.pdf']))
