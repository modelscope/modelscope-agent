import os
from typing import Any, Dict, List, Union

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.readers.base import BaseReader


class Knowledge(BaseLlamaPack):
    """ rag pipeline.

    从不同的源加载知识，支持：文件夹路径（str），文件路径列表（list），将不同源配置到不同的召回方式（dict）.
    Automatically select the best file reader given file extensions.

    Args:
        knowledge_source: Path to the directory，或文件路径列表，或指定召回方式的文件路径。
        cache_dir: 缓存indexing后的信息。
    """

    def __init__(self,
                 knowledge_source: Union[List, str, Dict],
                 cache_dir: str = './run',
                 **kwargs) -> None:

        # extra_readers = self.get_extra_readers()
        self.documents = []
        if isinstance(knowledge_source, str):
            if os.path.exists(knowledge_source):
                self.documents.append(
                    SimpleDirectoryReader(
                        input_dir=knowledge_source,
                        recursive=True).load_data())

        self.documents = SimpleDirectoryReader(
            input_files=knowledge_source).load_data()

    def get_extra_readers(self) -> Dict[str, BaseReader]:
        return {}

    def get_modules(self) -> Dict[str, Any]:
        """Get modules for rewrite."""
        return {
            'node_parser': self.node_parser,
            'recursive_retriever': self.recursive_retriever,
            'query_engines': self.query_engines,
            'reader': self.path_reader,
        }

    def run(self, query: str, **kwargs) -> str:
        return self.query_engine.query(query, **kwargs)
