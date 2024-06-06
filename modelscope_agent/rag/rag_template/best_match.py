from typing import Any, List, Optional

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from modelscope_agent.llm import get_chat_model
from modelscope_agent.rag.knowledge import BaseKnowledge


class BestMatchKnowledge(BaseKnowledge):

    def get_root_retriever(self,
                           documents: List[Document],
                           cache_dir: str,
                           llm: LLM,
                           chunk_size: int = 200,
                           similarity_top_k=2,
                           **kwargs) -> BaseRetriever:
        from llama_index.retrievers.bm25 import BM25Retriever

        self.splitter = SentenceSplitter(chunk_size=chunk_size)
        nodes = self.splitter.get_nodes_from_documents(documents)

        # initialize storage context (by default it's in-memory)
        storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
        storage_context.docstore.add_documents(nodes)
        if cache_dir is not None:
            storage_context.persist(persist_dir=cache_dir)

        # We can pass in the index, doctore, or list of nodes to create the retriever
        return BM25Retriever.from_defaults(
            docstore=storage_context.docstore,
            similarity_top_k=similarity_top_k)


if __name__ == '__main__':
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
    llm = get_chat_model(**llm_config)

    knowledge = BestMatchKnowledge('./data/常见QA.pdf', llm=llm)

    print(knowledge.run('如何创建agent', files=[]))
    print('-----------------------')
