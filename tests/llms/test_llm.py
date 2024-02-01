import pytest
from modelscope_agent.llm import QwenChatAtDS

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
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': 'test'
    }
    chat_model = QwenChatAtDS(**llm_config)
    mocker.patch.object(
        chat_model, '_chat_stream', return_value=['hello', ' there'])
    mocker.patch.object(
        chat_model, '_chat_no_stream', return_value='hello there')
    return chat_model


def test_chat_no_stream_with_prompt(chat_model):
    response = chat_model.chat(prompt=prompt)
    assert isinstance(response, str)
    assert response.strip() == 'hello there'


def test_chat_no_stream_with_messages(chat_model):
    response = chat_model.chat(messages=messages)
    assert isinstance(response, str)
    assert response.strip() == 'hello there'


def test_chat_stream_with_prompt(chat_model):
    responses = list(chat_model.chat(prompt=prompt, stream=True))
    assert isinstance(responses, list)
    assert all(isinstance(resp, str) for resp in responses)
    assert all(resp.strip() for resp in responses)
    assert responses == ['hello', ' there']


def test_chat_no_stream_with_invalid_messages(chat_model):
    with pytest.raises(
            AssertionError, match='messages list must not be empty'):
        chat_model.chat(messages=[])
