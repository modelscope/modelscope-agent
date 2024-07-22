# Retrieve with langchain
(deprecating)

## Long-term memory with Langchain VectorStore

We have already applied short-term memory in agent through simply concatenation of history. However, long-term memory can also be introduced through `langchain.vectorstores`. Here is a simple example.

In this example, we build a local vector store with some local documents. With each task, a retrieval in vector store will be executed first to get relevant information. And this information will be concated with origin query and feed to LLM.


### Initialize agent

This step is the same as other tutorials.
```Python
# initialize agent
...

agent = AgentExecutor(llm, tool_cfg, additional_tool_list=additional_tool_list, prompt_generator=prompt_generator)

```

### Build vector stores

Then we will use `langchain` components to build local vector stores.

First, We need to specify the embedding to be used. `langchain` have already supported both `ModelScopeEmbeddings` and `DashScopeEmbeddings`

```Python
# modelscope embeddings
model_id = 'damo/nlp_corom_sentence-embedding_chinese-base'
embeddings = ModelScopeEmbeddings(model_id=model_id)

# dashscope embeddings
embeddings = DashScopeEmbeddings(dashscope_api_key="my-api-key", model="text-embedding-v1")

```

Then we should load local files and use `TextSplitter` to split files into sentences. Here we use `damo/nlp_bert_document-segmentation_chinese-base` model in **ModelScope** as `TextSplitter`.

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

With embedding and splitting sentences, we can use `FAISS` to build index for further retrieval.

```Python
filepaths = ['tmp/ms.txt', 'tmp/china.txt', 'tmp/xiyouji.txt']

# load split file, transform to embedding
docs = load_file(filepaths)

# build index
vector_store = FAISS.from_documents(docs, embeddings)
```

### Rewrite query with retrieval


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



## Gradio demo of ModelScopeGPT
