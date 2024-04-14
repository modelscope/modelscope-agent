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
    # 1. test get history token count
    history_token_count_1 = memory.get_history_token_count()
    assert isinstance(history_token_count_1, int)

    # 2. test update history
    msg = [
        Message(role='system', content='test token counting'),
        Message(role='user', content='test token counting'),
    ]
    memory.update_history(msg)
    history_token_count_2 = memory.get_history_token_count()
    assert isinstance(history_token_count_2, int)
    assert history_token_count_2 > history_token_count_1

    # 3. test pop history
    memory.pop_history()
    history_token_count_3 = memory.get_history_token_count()
    assert isinstance(history_token_count_3, int)
    assert history_token_count_2 > history_token_count_3 > 0

    # 4. test clear history
    memory.clear_history()
    history_token_count_4 = memory.get_history_token_count()
    assert isinstance(history_token_count_4, int)
    assert history_token_count_4 == 0
