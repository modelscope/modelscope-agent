import json
from langchain_community.embeddings import ModelScopeEmbeddings
from modelscope_agent_servers.assistant_server.models import ChatResponse


class EmbeddingSingleton:
    _instance = None
    _is_initialized = False  # 初始化标志

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingSingleton,
                                  cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self._is_initialized = True
            self.embedding = ModelScopeEmbeddings(
                model_id='damo/nlp_gte_sentence-embedding_chinese-base')

    def get_embedding(self):
        return self.embedding


def tool_calling_wrapper(response: ChatResponse):
    """
    output should be in the format of openai choices
    "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [
              {
                "id": "call_abc123",
                "type": "function",
                "function": {
                  "name": "get_current_weather",
                  "arguments": "{\n\"location\": \"Boston, MA\"\n}"
                }
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ],

    Args:
        response: the chatresponse object

    Returns: dict

    """
    # TODO: only support one tool call for now
    response_dict = response.dict()
    choices = [{
        'index': 0,
        'message': {
            'role': 'assistant',
            'content': response_dict['response'],
        }
    }]
    if response_dict['tool'] is not None:
        choices[0]['message']['tool_calls'] = [{
            'type': 'function',
            'function': {
                'name':
                response_dict['tool']['name'],
                'arguments':
                json.dumps(
                    response_dict['tool']['inputs'], ensure_ascii=False)
            }
        }]
        choices[0]['finish_reason'] = 'tool_calls'
    else:
        choices[0]['finish_reason'] = 'stop'
    return choices
