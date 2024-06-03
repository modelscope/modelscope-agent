import os
from typing import Any, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import Document, TransformComponent
from llama_index.core.settings import Settings
from modelscope_agent.llm import get_chat_model
from modelscope_agent.rag.emb.dashscope import DashscopeEmbedding
from modelscope_agent.rag.knowledge import BaseKnowledge


class FusionKnowledge(BaseKnowledge):

    def get_root_retriever(self,
                           documents: List[Document],
                           cache_dir: str,
                           transformations: Optional[List[TransformComponent]],
                           chunk_size: int = 200,
                           similarity_top_k=2,
                           **kwargs) -> BaseRetriever:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever

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
        vector_retriever = index.as_retriever()
        bm_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=similarity_top_k)

        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm_retriever],
            llm=llm,
            num_queries=0)


if __name__ == '__main__':
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
    llm = get_chat_model(**llm_config)

    knowledge = FusionKnowledge('./data/常见QA.pdf', llm=llm)

    print(knowledge.run('如何创建agent', files=[]))
    print('-----------------------')
