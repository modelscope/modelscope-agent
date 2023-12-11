import os
from typing import Dict, Iterable, List, Union

import json
from langchain.document_loaders import (PyPDFLoader, TextLoader,
                                        UnstructuredFileLoader)
from langchain.embeddings import ModelScopeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore


class Retrieval:

    def __init__(self,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        self.embedding = embedding or ModelScopeEmbeddings(
            model_id='damo/nlp_gte_sentence-embedding_chinese-base')
        self.top_k = top_k
        self.vs_cls = vs_cls or FAISS
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
        if 'page' in res[0].metadata:
            res.sort(key=lambda doc: doc.metadata['page'])
        return [r.page_content for r in res]


class ToolRetrieval(Retrieval):

    def __init__(self,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
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
                 docs,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        super().__init__(embedding, vs_cls, top_k, vs_params)
        self.construct(docs)

    @staticmethod
    def file_preprocess(file_path):
        textsplitter = CharacterTextSplitter()
        all_files = []
        if isinstance(file_path, str) and os.path.isfile(file_path):
            all_files.append(file_path)
        elif isinstance(file_path, list):
            all_files = file_path
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
            elif f.lower().endswith('.pdf'):
                loader = PyPDFLoader(f)
                docs += (loader.load_and_split(textsplitter))
            else:
                raise ValueError(
                    f'not support file type: {f}, will be support soon')
        return docs

    @classmethod
    def from_file(cls,
                  file_path: Union[str, list],
                  embedding: Embeddings = None,
                  vs_cls: VectorStore = None,
                  top_k: int = 5,
                  vs_params: Dict = {}):
        # default embedding and vs class
        if embedding is None:
            model_id = 'damo/nlp_gte_sentence-embedding_chinese-base'
            embedding = ModelScopeEmbeddings(model_id=model_id)
        if vs_cls is None:
            vs_cls = FAISS
        docs = KnowledgeRetrieval.file_preprocess(file_path)

        if len(docs) == 0:
            return None
        else:
            return cls(docs, embedding, vs_cls, top_k, vs_params)

    def add_file(self, file_path: Union[str, list]):
        docs = KnowledgeRetrieval.file_preprocess(file_path)
        self.add_docs(docs)

    def add_docs(self, docs):
        assert len(docs) > 0
        if isinstance(docs[0], str):
            self.vs.add_texts(docs, **self.vs_params)
        elif isinstance(docs[0], Document):
            self.vs.add_documents(docs, **self.vs_params)
