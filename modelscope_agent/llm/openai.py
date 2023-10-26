import os

import openai

from .base import LLM

openai.api_key = os.getenv('OPENAI_API_KEY')


class OpenAi(LLM):
    name = 'openai'

    def __init__(self, cfg):
        super().__init__(cfg)

        self.model = self.cfg.get('model', 'gpt-3.5-turbo')
        self.api_base = self.cfg.get('api_base', 'https://api.openai.com/v1')

    def generate(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]

        response = openai.ChatCompletion.create(
            model=self.model,
            api_base=self.api_base,
            messages=messages,
            stream=False)
        completions = []
        if response and 'choices' in response:
            for choice in response['choices']:
                if 'message' in choice:
                    completions.append(choice['message']['content'])
        response = ''.join(completions)

        # truncate response
        response = response.split('<|user|>')[0]
        idx = response.find('<|endofthink|>')
        if idx != -1:
            response = response[:idx + len('<|endofthink|>')]
        return response
