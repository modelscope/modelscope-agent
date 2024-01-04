import os
from typing import Dict, Iterator, List, Optional

import json
import requests

from .base import BaseChatModel, register_llm


@register_llm('custom')
class CustomLLM(BaseChatModel):
    '''
        the llm service from http.
    '''

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)

        self.token = os.getenv('HTTP_LLM_TOKEN', None)
        self.model = os.getenv('HTTP_LLM_MODEL', None)
        self.url = os.getenv('HTTP_LLM_URL', None)

        if self.token is None:
            raise ValueError('HTTP_LLM_TOKEN is not set')

    def http_request(self, data):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        response = requests.post(self.url, json=data, headers=headers)
        return json.loads(response.content)

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:

        data = {'model': self.model, 'messages': messages, 'n': 1}
        message = self._inference(data)
        return message['content']

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        # todo: implement the streaming chat
        raise NotImplementedError

    def chat_with_functions(self,
                            messages: List[Dict],
                            functions: Optional[List[Dict]] = None,
                            **kwargs) -> Dict:

        data = {
            'model': self.model,
            'messages': messages,
            'n': 1,
            'functions': functions,
            'function_call': kwargs.get('function_call', 'auto')
        }

        message = self._inference(data)

        new_message = {
            'content': message['content'],
            'role': message.get('response_role', 'assistant')
        }
        if 'function_call' in message and message['function_call'] != {}:
            new_message['function_call'] = message.get('function_call')

        return new_message

    def _inference(self, data: dict) -> dict:
        retry_count = 0
        max_retries = 3
        message = {'content': ''}
        while retry_count <= max_retries:
            try:
                response = self.http_request(data)
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    import traceback
                    traceback.print_exc()
                    print(f'input: {data}, original error: {str(e)}')
                    raise e

            if response['code'] == 200:
                message = response['data']['response'][0]
                break
            else:
                retry_count += 1
                if retry_count > max_retries:
                    print('maximum retry reached, return default message')
        return message
