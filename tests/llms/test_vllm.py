from unittest.mock import MagicMock

import pytest
from modelscope_agent.llm import OpenAi

prompt = 'Tell me a joke.'
messages = [{
    'role': 'user',
    'content': 'Hello.'
}, {
    'role': 'assistant',
    'content': 'Hi there!'
}, {
    'role': 'user',
    'content': 'Tell me a joke.'
}]


@pytest.fixture
def chat_model(mocker):
    # using mock llm result as output
    llm_config = {
        'model': 'qwen',
        'model_server': 'openai',
        'api_base': 'http://127.0.0.1:8000/v1',
        'api_key': 'EMPTY'
    }
    chat_model = OpenAi(**llm_config)
    mocker.patch.object(
        chat_model, '_chat_stream', return_value=['hello', ' there'])
    mocker.patch.object(
        chat_model, '_chat_no_stream', return_value='hello there')
    return chat_model


@pytest.fixture
def chat_model_usage(mocker):
    """
    Fixture to create a mock chat model for testing usage info without pre-patching.
    """
    llm_config = {
        'model': 'qwen',
        'model_server': 'openai',
        'api_base': 'http://127.0.0.1:8000/v1',
        'api_key': 'EMPTY'
    }
    chat_model = OpenAi(**llm_config)
    return chat_model


def test_chat_stop_word(chat_model):
    stop = chat_model._update_stop_word(['observation'])
    assert isinstance(stop, list)
    assert stop == ['<|im_end|>', 'observation']
    stop = chat_model._update_stop_word(None)
    assert isinstance(stop, list)
    assert stop == ['<|im_end|>']
    stop = chat_model._update_stop_word([])
    assert isinstance(stop, list)
    assert stop == ['<|im_end|>']


def test_chat_no_stream_usage_info(chat_model_usage, mocker):
    """
    Test the _chat_no_stream method to ensure it updates the usage info correctly.
    """
    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = 'Test content'
    mock_response.usage.dict.return_value = {
        'prompt_tokens': 5,
        'completion_tokens': 10,
        'total_tokens': 15
    }

    # Patch the create method of the client to return the mock response
    mocker.patch.object(
        chat_model_usage.client.chat.completions,
        'create',
        return_value=mock_response)

    messages = [{'role': 'user', 'content': 'Hello!'}]

    # Call the method
    result = chat_model_usage._chat_no_stream(messages)

    # Check if the method returns the correct content
    assert result == 'Test content'

    # Check if the usage info is correctly updated
    assert chat_model_usage.last_call_usage_info == {
        'prompt_tokens': 5,
        'completion_tokens': 10,
        'total_tokens': 15
    }
