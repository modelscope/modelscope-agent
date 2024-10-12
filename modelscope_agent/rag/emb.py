import os
from abc import abstractmethod
from enum import Enum
from http import HTTPStatus
from typing import Any, List, Optional

import dashscope
from llama_index.core.base.embeddings.base import BaseEmbedding

# Enums for validation and type safety
DashscopeModelName = [
    'text-embedding-v1',
    'text-embedding-v2',
]


# Assuming BaseEmbedding is a Pydantic model and handles its own initializations
class Embedding(BaseEmbedding):
    """DashscopeEmbedding uses the dashscope API to generate embeddings for text."""

    @abstractmethod
    def _embed(self,
               texts: List[str],
               text_type='document') -> List[List[float]]:
        """Embed sentences."""

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._embed([query], text_type='query')[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._embed([text], text_type='document')[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._embed(texts, text_type='document')


class DashscopeEmbedding(Embedding):
    """DashscopeEmbedding uses the dashscope API to generate embeddings for text."""

    def __init__(
        self,
        model_name: str = 'text-embedding-v2',
    ):
        """
        A class representation for generating embeddings using the dashscope API.

        Args:
            model_name (str): The name of the model to be used for generating embeddings. The class ensures that
                          this model is supported and that the input type provided is compatible with the model.
        """

        # Validate model_name and input_type
        if model_name not in DashscopeModelName:
            raise ValueError(f'model {model_name} is not supported.')

        super().__init__(model_name=model_name)

    @classmethod
    def class_name(cls) -> str:
        return 'DashscopeEmbedding'

    def _embed(self,
               texts: List[str],
               text_type='document') -> List[List[float]]:
        """Embed sentences using dashscope."""
        assert os.environ.get(
            'DASHSCOPE_API_KEY',
            None), 'DASHSCOPE_API_KEY should be set in environ.'
        resp = dashscope.TextEmbedding.call(
            input=texts,
            model=self.model_name,
            text_type=text_type,
        )
        if resp.status_code == HTTPStatus.OK:
            res = resp.output['embeddings']
        else:
            raise ValueError(f'call dashscope api failed: {resp}')

        return [list(map(float, e['embedding'])) for e in res]


if __name__ == '__main__':
    # Example usage
    embedding = DashscopeEmbedding(model_name='text-embedding-v2')
    query = 'This is a query'
    text = 'This is a document'
    query_embedding = embedding._embed(query)
    print(query_embedding)
