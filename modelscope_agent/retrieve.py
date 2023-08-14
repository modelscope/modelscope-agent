import os
from typing import Dict, Iterable, List

import json
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore


class Retrieval:

    def __init__(self,
                 embedding: Embeddings,
                 vs_cls: VectorStore,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        self.embedding = embedding
        self.top_k = top_k
        self.vs_cls = vs_cls
        self.vs_params = vs_params
        self.vs = None

    def construct(self, docs):
        assert len(docs) > 0
        if isinstance(docs[0], str):
            self.vs = self.vs_cls.from_texts(docs, self.embedding,
                                             **self.vs_params)
        elif isinstance(docs[0], Document):
            self.vs = self.vs_cls.from_documents(docs, self.embedding,
                                                 **self.vs_params)

    def retrieve(self, query: str) -> List[str]:
        res = self.vs.similarity_search(query, k=self.top_k)
        return [r.page_content for r in res]


class ToolRetrieval(Retrieval):

    def __init__(self,
                 embedding: Embeddings,
                 vs_cls: VectorStore,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        super().__init__(embedding, vs_cls, top_k, vs_params)

    def retrieve(self, query: str) -> Dict[str, str]:
        res = self.vs.similarity_search(query, k=self.top_k)

        final_res = {}

        for r in res:
            content = r.page_content
            name = json.loads(content)['name']
            final_res[name] = content

        return final_res


class KnowledgeRetrieval(Retrieval):

    def __init__(self,
                 embedding: Embeddings,
                 vs_cls: VectorStore,
                 docs,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        super().__init__(embedding, vs_cls, top_k, vs_params)
        self.construct(docs)

    @classmethod
    def from_file(cls,
                  embedding: Embeddings,
                  vs_cls: VectorStore,
                  file_path: str,
                  top_k: int = 5,
                  vs_params: Dict = {}):

        textsplitter = CharacterTextSplitter()
        all_files = []
        if os.path.isfile(file_path):
            all_files.append(file_path)
        elif os.path.isdir(file_path):
            for root, dirs, files in os.walk(file_path):
                for f in files:
                    all_files.append(os.path.join(root, f))
        else:
            raise ValueError('file_path must be a file or a directory')

        docs = []
        for f in all_files:
            if f.lower().endswith('.txt'):
                loader = TextLoader(f, autodetect_encoding=True)
                docs += (loader.load_and_split(textsplitter))
            elif f.lower().endswith('.md'):
                loader = UnstructuredFileLoader(f, mode='elements')
                docs += loader.load()

        return cls(embedding, vs_cls, docs, top_k, vs_params)
