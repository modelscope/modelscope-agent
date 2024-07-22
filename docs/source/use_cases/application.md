# 基于轻量化RAG的应用示例
## 使用 Langchain VectorStore 实现长期记忆
（deprecating）
我们已经通过简单地拼接历史记录在agent中实现了短期记忆。然而，长期记忆也可以通过`langchain.vectorstores`引入。这里是一个简单的例子。

在这个例子中，我们使用一些本地文档建立了一个本地向量存储。对于每个任务，首先会在向量存储中执行检索以获取相关信息。然后，这些信息将与原始查询拼接并输入到 LLM（大型语言模型）。

### 初始化agent
这一步骤与其他教程相同。

```Python
# initialize agent
...

agent = AgentExecutor(llm, tool_cfg, additional_tool_list=additional_tool_list, prompt_generator=prompt_generator)

```

### 构建向量存储
接下来我们将使用`langchain`组件来构建本地向量存储。

首先，我们需要指定要使用的嵌入。 `langchain`已经支持`ModelScopeEmbeddings`和`DashScopeEmbeddings`。

```Python
# modelscope embeddings
model_id = 'damo/nlp_corom_sentence-embedding_chinese-base'
embeddings = ModelScopeEmbeddings(model_id=model_id)

# dashscope embeddings
embeddings = DashScopeEmbeddings(dashscope_api_key="my-api-key", model="text-embedding-v1")

```

然后我们应该加载本地文件，并使用`TextSplitter`将文件按句子分割。这里我们在**ModelScope**中使用`damo/nlp_bert_document-segmentation_chinese-base`模型作为`TextSplitter`。

```Python
# ref: https://github.com/chatchat-space/langchain-ChatGLM/blob/master/textsplitter/ali_text_splitter.py
class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:

        p = pipeline(
            task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list

def load_file(filepaths, sentence_size=100):
    textsplitter = AliTextSplitter()
    docs = []
    for filepath in filepaths:
        if not filepath.lower().endswith(".txt"):
            continue
        loader = TextLoader(filepath, autodetect_encoding=True)
        docs+=(loader.load_and_split(textsplitter))

    return docs

```

通过嵌入和分割的句子，我们可以使用`FAISS`来构建索引，以便进行进一步的检索。

```Python
filepaths = ['tmp/ms.txt', 'tmp/china.txt', 'tmp/xiyouji.txt']

# load split file, transform to embedding
docs = load_file(filepaths)

# build index
vector_store = FAISS.from_documents(docs, embeddings)
```

### 使用检索重写查询

```Python
top_k = 3
def search_query_wrapper(query):

    search_docs = vector_store.similarity_search(query, k=top_k)

    search_res = '\n'.join([f'[{idx+1}] {s.page_content}' for idx, s in enumerate(search_docs)])

    final_query = f'<|startofsearch|>\nsearch results: \n{search_res.strip()}\n<|endofsearch|>\n{query}'

    return final_query

# origin query
query_without_search = '魔搭安装'
# rewrite query
query_with_search = search_query_wrapper(query_without_search)
```
