# BM25 停用词表(vendored)

来源:HuggingFace 仓库 `Qdrant/bm25`(Apache-2.0),即 fastembed 的
`SparseTextEmbedding("Qdrant/bm25")` 运行所需的全部文件——它不是神经网络模型,
只是 Snowball 系停用词表;`mock.file` 从不被读取。2026-08-19 经 hf-mirror 抓取,
逐文件检视为纯词表。`tamil.txt` 上游仓库缺失(fastembed 原生下载同样拿不到,
行为等价)。

vendor 的原因:mem0 的 qdrant store 用它编码混合检索的稀疏向量,而该仓库在
ModelScope 没有镜像——不带上它,HuggingFace 不可达的网络上混合检索会静默降级为
纯稠密。`embed_warmup.warm_bm25_cache` 把这些文件按 huggingface_hub 布局铺进
fastembed 缓存,使其完全离线可用。
