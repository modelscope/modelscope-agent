import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

import fsspec
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, QueryBundle, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (MetadataFilter,
                                                  MetadataFilters)
from llama_index.legacy.core.embeddings.base import BaseEmbedding
from modelscope_agent.llm import get_chat_model
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.rag.emb import DashscopeEmbedding
from modelscope_agent.rag.llm import ModelscopeAgentLLM
from modelscope_agent.utils.nltk_utils import install_nltk_data

install_nltk_data()


@dataclass
class FileQueryBundle(QueryBundle):
    files: List[str] = None


# @register_rag('base_knowledge')
class BaseKnowledge(BaseLlamaPack):
    """ base knowledge pipeline.

    Better use of knowledge base content through LLM.
    Automatically select the best file reader given file extensions.

    Args:
        files: Path to the directory, or list of file_paths, defaults to empty list.
        cache_dir: Directory to cache indexed content, defaults to `./run`.
        llm: Language model is used to summarize retrieved content, defaults to Dashscope qwen-max.
        retriever: The retriever strategies. It should be a subclass of llama-index BaseRetriever. The default
            class is VectorIndexRetriever.
        loaders: Additional file Readers. The parameter format is a dictionary mapping file extensions to
            Reader classes. The reader classes should be subclasses of llama-index BaseReader. The file types
            that already have corresponding readers are: `.hwp`, `.pdf`, `.docx`, `.pptx`, `.ppt`, `.pptm`,
            `.jpg`, `.png`, `.jpeg`, `.mp3`, `.mp4`, `.csv`, `.epub`, `.md`, `.mbox`, `.ipynb`, `txt`, `.pd`,
            `.html`.
        transformations: The chunk or split strategies. It should be a subclass of llama-index TransformComponent.
            The default is SentenceSplitter.
        post_processors: The processors of retrieved contents, such of re-rank. The default is None.
    """

    def __init__(self,
                 files: Union[List, str] = [],
                 cache_dir: str = './run',
                 llm: Union[LLM, BaseChatModel, Dict] = {},
                 retriever: Optional[Type[BaseRetriever]] = None,
                 emb: Optional[Type[BaseEmbedding]] = None,
                 loaders: Dict[str, Union[BaseReader, Type[BaseReader]]] = {},
                 transformations: List[Type[TransformComponent]] = [],
                 post_processors: List[Type[BaseNodePostprocessor]] = [],
                 use_cache: bool = True,
                 **kwargs) -> None:
        self.retriever_cls = retriever
        self.cache_dir = cache_dir
        # self.register_files(files) # TODO: file manager
        self.extra_readers = self.get_extra_readers(loaders)
        self.embed_model = self.get_emb_model(emb)
        Settings._embed_model = self.embed_model
        documents = None
        if not use_cache:
            documents = self.read(files)

        self.llm = self.get_llm(llm)
        Settings._llm = self.llm

        # 可对本召回器的文本范围 进行过滤、筛选、rechunk。transformations为空时，默认按语义rechunk。
        self.transformations = self.get_transformations(transformations)

        self.postprocessors = self.get_postprocessors(post_processors,
                                                      **kwargs)

        root_retriever = self.get_root_retriever(
            documents, use_cache=use_cache, **kwargs)

        if root_retriever:
            self.query_engine = self.get_query_engine(root_retriever, **kwargs)

    def get_llm(self, llm: Union[LLM, BaseChatModel, Dict]) -> LLM:
        llama_index_llm = None
        if llm and isinstance(llm, BaseChatModel):
            llama_index_llm = ModelscopeAgentLLM(llm)
        elif isinstance(llm, LLM):
            llama_index_llm = llm
        elif isinstance(llm, dict):
            try:
                ms_agent_llm = get_chat_model(**llm)
                llama_index_llm = ModelscopeAgentLLM(ms_agent_llm)
            except Exception as e:
                print(
                    f'Unable to initialize llm throuth {llm}, using dashscope:qwen-max instead. Failed reason: {e}'
                )
        elif llm:
            print(
                f'Unsupported parameter type: llm={llm}. Expecting llama_index.core.llms.LLM,'
                'modelscope_agent.llm.base.BaseChatModel or llm_config from modelscope_agent'
            )

        if not llama_index_llm:
            llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
            ms_agent_llm = get_chat_model(**llm_config)
            llama_index_llm = ModelscopeAgentLLM(ms_agent_llm)
        return llama_index_llm

    def get_emb_model(self,
                      emb_cls: Optional[Type[BaseEmbedding]]) -> BaseEmbedding:
        emb_model = None
        if emb_cls:
            try:
                emb_model = emb_cls()
            except Exception as e:
                print(
                    f'Unable to initialize {emb_cls}, using dashscope embedding instead. Details:{e}'
                )
        if not emb_model:
            emb_model = DashscopeEmbedding()
        return emb_model

    def get_query_engine(self, root_retriever: BaseRetriever,
                         **kwargs) -> BaseQueryEngine:
        return RetrieverQueryEngine.from_args(
            root_retriever,
            llm=self.llm,
            node_postprocessors=self.postprocessors)

    def get_transformations(self,
                            transformations: List[Type[TransformComponent]],
                            **kwargs) -> Optional[List[TransformComponent]]:
        # rechunk，筛选文档内容等
        res = []
        for t_cls in transformations:
            try:
                t = t_cls()
                res.append(t)
            except Exception as e:
                print(
                    f'node parser {t_cls} cannot be used and it will be ignored. Detail: {e}'
                )
        return res

    def get_postprocessors(
            self, post_processors: List[Type[BaseNodePostprocessor]],
            **kwargs) -> Optional[List[Type[BaseNodePostprocessor]]]:
        # 获取召回内容后处理器
        res = []
        for post_processor_cls in post_processors:
            try:
                post_processor = post_processor_cls()
                res.append(post_processor)
            except Exception as e:
                print(
                    f'post_processor_cls {post_processor_cls} cannot be used and it will be ignored. Detail: {e}'
                )

        return res

    def get_root_retriever(self,
                           documents: List[Document],
                           use_cache: bool = True,
                           **kwargs) -> BaseRetriever:

        # indexing
        # 可配置chunk_size等
        Settings.chunk_size = 512
        index = None
        if use_cache:
            if self.cache_dir is not None and os.path.exists(self.cache_dir):
                try:
                    # Load from cache
                    from llama_index.core import StorageContext, load_index_from_storage
                    # rebuild storage context
                    storage_context = StorageContext.from_defaults(
                        persist_dir=self.cache_dir)
                    # load index

                    index = load_index_from_storage(
                        storage_context, embed_model=self.embed_model)
                except Exception as e:
                    print(
                        f'Can not load index from cache_dir {self.cache_dir}, detail: {e}'
                    )

        if documents is not None:
            if not index:
                index = VectorStoreIndex.from_documents(
                    documents=documents,
                    transformations=self.transformations,
                    embed_model=self.embed_model)
            else:
                for doc in documents:
                    index.insert(doc)
        if not index:
            print('Neither documents nor cache_dir.')
            return None

        if self.cache_dir is not None:
            index.storage_context.persist(persist_dir=self.cache_dir)

        # init retriever tool
        if self.retriever_cls:
            try:
                return self.retriever_cls(index)
            except Exception as e:
                print(
                    f'Retriever {self.retriever_cls} cannot be used, using default retriever instead. Detail: {e}'
                )

        return index.as_retriever()

    def get_extra_readers(
        self, loaders: Dict[str, Union[BaseReader, Type[BaseReader]]]
    ) -> Dict[str, BaseReader]:
        extra_readers = {}
        for file_type, loader_or_cls in loaders.items():
            if isinstance(loader_or_cls, BaseReader):
                extra_readers[file_type] = loader_or_cls
            try:
                loader = loader_or_cls()
                extra_readers[file_type] = loader
            except Exception as e:
                print(
                    f'Using {loader_or_cls} failed. Can not read {file_type} file. Detail: {e}'
                )

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

    def read(self,
             knowledge_source: Union[str, List[str]],
             exclude_hidden: bool = True,
             recursive: bool = False,
             fs: Optional[fsspec.AbstractFileSystem] = None,
             **kwargs) -> List[Document]:
        documents = []
        try:
            if isinstance(knowledge_source, str):
                if os.path.isdir(knowledge_source):
                    general_reader = SimpleDirectoryReader(
                        input_dir=knowledge_source,
                        file_extractor=self.extra_readers,
                        exclude_hidden=exclude_hidden,
                        fs=fs,
                        recursive=recursive)
                elif os.path.isfile(knowledge_source):
                    general_reader = SimpleDirectoryReader(
                        input_files=[knowledge_source],
                        file_extractor=self.extra_readers,
                        exclude_hidden=exclude_hidden,
                        fs=fs,
                        recursive=recursive)
                else:
                    raise ValueError(
                        f'file path not exists: {knowledge_source}.')
            else:
                general_reader = SimpleDirectoryReader(
                    input_files=knowledge_source,
                    file_extractor=self.extra_readers,
                    fs=fs,
                    exclude_hidden=exclude_hidden,
                    recursive=recursive)

            # documents = general_reader.load_data(num_workers=os.cpu_count())
            documents = general_reader.load_data()
        except ValueError as e:
            print(f'No valid documents, {e}')
            documents = []
        return documents

    def set_filter(self, files: List[str]):
        retriever = self.query_engine.retriever
        filters = [
            MetadataFilter(key='file_name', value=os.path.basename(file))
            for file in files
        ]
        retriever._filters = MetadataFilters(filters=filters)

    def run(self, query: str, files: List[str] = [], **kwargs) -> str:
        query_bundle = FileQueryBundle(query)
        if isinstance(files, str):
            files = [files]

        if files and len(files) > 0:
            self.set_filter(files)

        return str(self.query_engine.query(query_bundle, **kwargs))

    def add(self, files: List[str]):
        if isinstance(files, str):
            files = [files]

        try:
            documents = self.read(files)
            root_retriever = self.get_root_retriever(documents, use_cache=True)
            self.query_engine = self.get_query_engine(root_retriever)

        except BaseException as e:
            print(f'add files {files} failed, detail: {e}')


if __name__ == '__main__':
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
    llm = get_chat_model(**llm_config)

    knowledge = BaseKnowledge('./data2', use_cache=False, llm=llm)

    knowledge.add(['./data/常见QA.pdf'])
    print(knowledge.run('高德天气API申请', files=['常见QA.pdf']))
