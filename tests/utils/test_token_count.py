import os

import pytest
from modelscope_agent.memory.memory_with_retrieval_knowledge import \
    MemoryWithRetrievalKnowledge
from modelscope_agent.schemas import Message

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
    old_token_count = memory.get_token_count()
    assert isinstance(old_token_count, int)
    msg = Message(role='system', content='test token counting')
    memory.update_history(msg)
    new_token_count = memory.get_token_count()
    assert isinstance(new_token_count, int)
    assert new_token_count > old_token_count
