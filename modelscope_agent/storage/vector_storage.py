import os
from pathlib import Path
from typing import Dict, List, Union

import json
from langchain.schema import Document
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.embeddings import Embeddings
from modelscope_agent.utils.parse_doc import parse_doc

from .base import BaseStorage

SUPPORTED_KNOWLEDGE_TYPE = ['txt', 'md', 'pdf', 'docx', 'pptx', 'md']


class VectorStorage(BaseStorage):

    def __init__(self,
                 storage_path: Union[str, Path],
                 index_name: str,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = FAISS,
                 vs_params: Dict = {},
                 index_ext: str = '.faiss',
                 use_cache: bool = True,
                 **kwargs):
        # index name used for storage
        self.storage_path = str(storage_path)
        self.index_name = index_name
        self.embedding = embedding or ModelScopeEmbeddings(
            model_id='damo/nlp_gte_sentence-embedding_chinese-base')
        self.vs_cls = vs_cls
        self.vs_params = vs_params
        self.index_ext = index_ext
        if use_cache:
            self.vs = self.load()
        else:
            self.vs = None

    def construct(self, docs):
        assert len(docs) > 0
        if isinstance(docs[0], str):
            self.vs = self.vs_cls.from_texts(docs, self.embedding,
                                             **self.vs_params)
        elif isinstance(docs[0], Document):
            self.vs = self.vs_cls.from_documents(docs, self.embedding,
                                                 **self.vs_params)

    def search(self, query: str, top_k=5) -> List[str]:
        if self.vs is None:
            return []
        res = self.vs.similarity_search(query, k=top_k)
        if 'page' in res[0].metadata:
            res.sort(key=lambda doc: doc.metadata['page'])
        return [r.page_content for r in res]

    def add(self, docs: Union[List[str], List[Document]]):
        assert len(docs) > 0
        if isinstance(docs[0], str):
            self.vs.add_texts(docs, **self.vs_params)
        elif isinstance(docs[0], Document):
            self.vs.add_documents(docs, **self.vs_params)

    def _get_index_and_store_name(self, index_ext='.index', pkl_ext='.pkl'):
        index_file = os.path.join(self.storage_path,
                                  f'{self.index_name}{index_ext}')
        store_file = os.path.join(self.storage_path,
                                  f'{self.index_name}{pkl_ext}')
        return index_file, store_file

    def load(self) -> Union[VectorStore, None]:
        if not self.storage_path or not os.path.exists(self.storage_path):
            return None
        index_file, store_file = self._get_index_and_store_name(
            index_ext=self.index_ext)

        if not (os.path.exists(index_file) and os.path.exists(store_file)):
            return None

        return self.vs_cls.load_local(
            self.storage_path,
            self.embedding,
            self.index_name,
            allow_dangerous_deserialization=True)

    def save(self):
        if self.vs:
            self.vs.save_local(self.storage_path, self.index_name)

    def delete(self):
        """Now, no delete is implemented"""
        raise NotImplementedError


class KnowledgeVector(VectorStorage):

    @staticmethod
    def file_preprocess(file_path: Union[str, List[str]]) -> List[Dict]:
        all_files = []
        if isinstance(file_path, str) and os.path.isfile(file_path):
            all_files.append(file_path)
        elif isinstance(file_path, list):
            for f in file_path:
                if os.path.isfile(f):
                    all_files.append(f)
        elif os.path.isdir(file_path):
            for root, dirs, files in os.walk(file_path):
                for f in files:
                    all_files.append(os.path.join(root, f))
        else:
            raise ValueError('file_path must be a file or a directory')

        docs = []
        for f in all_files:
            if f.split('.')[-1].lower() in SUPPORTED_KNOWLEDGE_TYPE:
                doc_list = parse_doc(f)
                if len(doc_list) > 0:
                    docs.extend(doc_list)
        return docs

    # should load and save
    def add(self, file_path: Union[str, list]):
        custom_docs = KnowledgeVector.file_preprocess(file_path)
        if len(custom_docs) > 0:
            text_docs = [docs['page_content'] for docs in custom_docs]

            if self.vs is None:
                self.construct(text_docs)
            else:
                super().add(text_docs)
