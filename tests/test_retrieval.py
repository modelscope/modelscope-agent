import os

import pytest
from langchain.embeddings import DashScopeEmbeddings, ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from modelscope_agent.retrieve import KnowledgeRetrieval, ToolRetrieval


@pytest.fixture
def initialized_tool_retrieval():
    # 开源版本的向量库配置
    model_id = 'damo/nlp_corom_sentence-embedding_chinese-base'
    embeddings = ModelScopeEmbeddings(model_id=model_id)
    file_path = os.path.join(
        os.path.dirname(__file__), 'samples', 'modelscope_qa_1.txt')
    knowledge_retrieval = KnowledgeRetrieval.from_file(file_path, embeddings,
                                                       FAISS, 1)
    return knowledge_retrieval


def test_base_knowledge_retrieval(initialized_tool_retrieval):
    assert initialized_tool_retrieval is not None
    query = 'ModelScope模型的需要联网么？'
    res = initialized_tool_retrieval.retrieve(query)
    assert len(res) == 1
    query = '支持算法评测么？'
    res1 = initialized_tool_retrieval.retrieve(query)
    assert len(res) == 1
    assert res[0] == res1[0]


def test_add_knowledge_retrieval(initialized_tool_retrieval):
    assert initialized_tool_retrieval is not None
    file_path = os.path.join(
        os.path.dirname(__file__), 'samples', 'modelscope_qa_2.txt')
    query = 'ModelScope模型的需要联网么？'
    res = initialized_tool_retrieval.retrieve(query)
    assert len(res) == 1

    initialized_tool_retrieval.add_file(file_path)

    query = '支持算法评测么？'
    res1 = initialized_tool_retrieval.retrieve(query)
    assert len(res) == 1
    assert res[0] != res1[0]
