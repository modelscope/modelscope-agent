import os

import pytest
from modelscope_agent.memory.memory_with_retrieval_knowledge import \
    MemoryWithRetrievalKnowledge

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)


@pytest.fixture
def temporary_storage(tmpdir):
    # Use a temporary directory for testing storage
    return str(tmpdir.mkdir('knowledge_vector_test'))


def test_memory_with_retrieval_knowledge(temporary_storage):
    random_name = 'test_memory_agent'

    memory = MemoryWithRetrievalKnowledge(
        storage_path=temporary_storage,
        name=random_name,
        memory_path=temporary_storage,
    )
    test_file = os.path.join(parent_dir, 'samples')

    # test add file to
    memory.run(query=None, url=test_file)
    assert os.path.exists(
        os.path.join(temporary_storage, random_name + '.faiss'))
    assert os.path.exists(
        os.path.join(temporary_storage, random_name + '.pkl'))

    result = memory.run(query='介绍memory', max_token=1000)
    assert isinstance(result, str)
    assert len(result) > 0 and len(result) < 1000
