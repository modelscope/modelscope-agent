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
