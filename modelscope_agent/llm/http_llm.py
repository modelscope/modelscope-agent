import os

import json
import requests

from .base import LLM


class HttpLLM(LLM):
    name = 'http_llm'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.token = os.getenv('HTTP_LLM_TOKEN', None)
        self.model = os.getenv('HTTP_LLM_model', None)
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

    def generate(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        data = {'model': self.model, 'messages': messages, 'n': 1}
        response = self.http_request(data)
        completions = []
        if response['code'] == 200:
            completions.append(response['data']['response'][0]['content'])
        response = ''.join(completions)

        # truncate response
        idx = response.find('<|endofthink|>')
        if idx != -1:
            response = response[:idx + len('<|endofthink|>')]
        return response
