from langchain_community.embeddings import ModelScopeEmbeddings


class EmbeddingSingleton:
    _instance = None
    _is_initialized = False  # 初始化标志

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingSingleton,
                                  cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self._is_initialized = True
            self.embedding = ModelScopeEmbeddings(
                model_id='damo/nlp_gte_sentence-embedding_chinese-base')

    def get_embedding(self):
        return self.embedding
