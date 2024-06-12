import os

import pytest
from modelscope_agent.storage import KnowledgeVector

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)


@pytest.fixture
def knowledge_vector_storage(tmpdir):
    # Use a temporary directory for testing storage
    temporary_storage = str(tmpdir.mkdir('knowledge_vector_test'))
    knowledge_vector = KnowledgeVector(
        storage_path=temporary_storage,
        index_name='test_index',
    )
    return knowledge_vector


def test_add_method_with_single_file(knowledge_vector_storage):

    # Define some test files
    test_file = os.path.join(parent_dir, 'samples', 'modelscope_qa_1.txt')

    # Add file to knowledge vectors
    knowledge_vector_storage.add(test_file)

    assert len(knowledge_vector_storage.vs.index_to_docstore_id) > 0


def test_add_method_with_dir(knowledge_vector_storage):
    # Define some test files
    test_file = os.path.join(parent_dir, 'samples')

    # Add file to knowledge vectors
    knowledge_vector_storage.add(test_file)

    assert len(knowledge_vector_storage.vs.index_to_docstore_id) > 0


def test_search_method(knowledge_vector_storage):

    # Define some test files
    test_file = os.path.join(parent_dir, 'samples', 'modelscope_qa_1.txt')

    # Add file to knowledge vectors
    knowledge_vector_storage.add(test_file)

    # Test the search method
    search_results = knowledge_vector_storage.search('介绍modelscope', top_k=1)
    assert isinstance(search_results, list)
    assert len(search_results) == 1

    search_results = knowledge_vector_storage.search('介绍modelscope', top_k=2)
    assert len(search_results) == 2

    # the chunk size could one be filled with 2
    search_results = knowledge_vector_storage.search('介绍modelscope', top_k=3)
    assert len(search_results) == 2


def test_save_and_load_methods(knowledge_vector_storage):

    # Test the save and load methods
    # Define some test files
    test_file = os.path.join(parent_dir, 'samples')

    # Add file to knowledge vectors
    knowledge_vector_storage.add(test_file)
    knowledge_vector_storage.save()
    storage_path = knowledge_vector_storage.storage_path

    # Perform assertions or additional tests as needed
    assert os.path.exists(os.path.join(storage_path, 'test_index.faiss'))
    assert os.path.exists(os.path.join(storage_path, 'test_index.pkl'))

    load_knowledge_vector = KnowledgeVector(
        storage_path=storage_path,
        index_name='test_index',
    )
    assert load_knowledge_vector.vs is not None
