# Retrieve模块

## 使用 Langchain VectorStore 实现长期记忆
我们已经通过简单地拼接历史记录实现了agent的短期记忆。然而，对于长期记忆，我们可以引入`langchain.vectorstores`和`langchain.embeddings`。这些组件被封装在`Retrieval`模块中。

在`Retrieval`模块中，我们从 Retrieval 类派生了两个类：

- `ToolRetrieval`：这个类用于工具检索。尽管工具的总数可能很多，但对于特定任务而言，只有3-5个工具可能是相关的。`ToolRetrieval`类负责筛选所有的工具。
- `KnowledgeRetrieval`：这个类用于构建本地知识库。对于某些任务，可能需要特定领域的相关知识。`KnowledgeRetrieval`类负责检索与给定任务相关的领域知识。


我们使用`DashScopeEmbeddings`和`FAISS`作为默认的嵌入和向量存储。但您可以轻松指定您想要使用的嵌入和向量存储。

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
