from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import dashscope
import pytest
from modelscope_agent.llm import DashScopeLLM, OllamaLLM, OpenAi


@pytest.fixture
def openai_chat_model_usage(mocker):
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


def test_openai_chat_no_stream_usage_info(openai_chat_model_usage, mocker):
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
        openai_chat_model_usage.client.chat.completions,
        'create',
        return_value=mock_response)

    messages = [{'role': 'user', 'content': 'Hello!'}]

    # Call the method
    result = openai_chat_model_usage._chat_no_stream(messages)

    # Check if the method returns the correct content
    assert result == 'Test content'

    # Check if the usage info is correctly updated
    assert openai_chat_model_usage.last_call_usage_info == {
        'prompt_tokens': 5,
        'completion_tokens': 10,
        'total_tokens': 15
    }


@pytest.fixture
def dashscope_chat_model(mocker):
    # using mock llm result as output
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'test'
    }
    chat_model = DashScopeLLM(**llm_config)
    return chat_model


def test_dashscope_chat_no_stream_usage_info(dashscope_chat_model, mocker):
    """
    Test the _chat_no_stream method to ensure it updates the usage info correctly.
    """
    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.output.choices = [MagicMock()]
    mock_response.output.choices[0].message.content = 'Test content'
    mock_response.usage.input_tokens = 5
    mock_response.usage.output_tokens = 10
    mock_response.usage.total_tokens = 15

    mock_response.status_code = HTTPStatus.OK

    # Patch the create method of the client to return the mock response
    mocker.patch.object(
        dashscope.Generation, 'call', return_value=mock_response)

    messages = [{'role': 'user', 'content': 'Hello!'}]

    # Call the method
    result = dashscope_chat_model._chat_no_stream(messages)

    # Check if the method returns the correct content
    assert result == 'Test content'

    # Check if the usage info is correctly updated
    assert dashscope_chat_model.last_call_usage_info['prompt_tokens'] == 5
    assert dashscope_chat_model.last_call_usage_info['completion_tokens'] == 10
    assert dashscope_chat_model.last_call_usage_info['total_tokens'] == 15


@pytest.fixture
def ollama_chat_model(mocker):
    # using mock llm result as output
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'ollama',
        'api_key': 'test'
    }
    chat_model = OllamaLLM(**llm_config)
    return chat_model


def test_ollama_chat_no_stream_usage_info(ollama_chat_model, mocker):
    """
    Test the _chat_no_stream method to ensure it updates the usage info correctly.
    """
    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.get.side_effect = lambda key, default=None: {
        'prompt_eval_count': 5,
        'eval_count': 10
    }.get(key, default)
    mock_response.__getitem__.side_effect = lambda key: {
        'message': {
            'content': 'Test content'
        }
    }[key]
    # Patch the create method of the client to return the mock response
    mocker.patch.object(
        ollama_chat_model.client, 'chat', return_value=mock_response)

    messages = [{'role': 'user', 'content': 'Hello!'}]

    # Call the method
    result = ollama_chat_model._chat_no_stream(messages)

    # Check if the method returns the correct content
    assert result == 'Test content'

    # Check if the usage info is correctly updated
    assert ollama_chat_model.last_call_usage_info['prompt_tokens'] == 5
    assert ollama_chat_model.last_call_usage_info['completion_tokens'] == 10
    assert ollama_chat_model.last_call_usage_info['total_tokens'] == 15
