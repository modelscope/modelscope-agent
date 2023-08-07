import openai

from .base import LLM


class OpenAi(LLM):
    name = 'openai'

    def __init__(self, cfg):
        super().__init__(cfg)

        self.model = self.cfg.get('model', '')
        self.api_key = self.cfg.get('api_key', '')
        self.api_base = self.cfg.get('api_base', '')

    def generate(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]

        response = openai.ChatCompletion.create(
            model=self.model,
            api_key=self.api_key,
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
        idx = response.find('<|endofthink|>')
        if idx != -1:
            response = response[:idx + len('<|endofthink|>')]
        return response
