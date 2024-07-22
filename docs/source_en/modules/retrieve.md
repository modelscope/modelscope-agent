# Retrieve Module
(deprecating)

## Long-term memory with Langchain VectorStore

We have implemented short-term memory in the agent by simply concatenating the history. However, for long-term memory, we can introduce `langchain.vectorstores` and `langchain.embeddings`. These components are encapsulated within the `Retrieval` module.

In `Retrieval` module, we have derived two classes from the Retrieval class:

- `ToolRetrieval`: This class is used for tool retrieval. While the total number of tools may be large, only 3-5 tools may be relevant to a specific task. The `ToolRetrieval` class is responsible for filter of all the tools.
- `KnowledgeRetrieval`: This class is used for constructing a local knowledge database. For certain tasks, there may be a need for relevant knowledge in specific domains. The `KnowledgeRetrieval` class is responsible for retrieving domain-specific knowledge related to a given task.

We use `DashScopeEmbeddings` and `FAISS` as default embedding and vector store. But you can easily specify the embedding and vectorstore you want to use.


```Python
class Retrieval:
    def __init__(self,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        self.embedding = embedding or DashScopeEmbeddings(
            model='text-embedding-v1')
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
```
