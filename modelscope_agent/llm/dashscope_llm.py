import os
from http import HTTPStatus

import dashscope
from dashscope import Generation

from .base import LLM

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')


class DashScopeLLM(LLM):
    name = 'dashscope_llm'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.cfg.get('model', 'modelscope-agent-llm-v1')
        self.generate_cfg = self.cfg.get('generate_cfg', {})

    def generate(self, prompt):

        total_response = ''
        responses = Generation.call(
            model=self.model, prompt=prompt, stream=False, **self.generate_cfg)

        if responses.status_code == HTTPStatus.OK:
            total_response = responses.output['text']
        else:
            print('Code: %d, status: %s, message: %s' %
                  (responses.status_code, responses.code, responses.message))

        idx = total_response.find('<|endofthink|>')
        if idx != -1:
            total_response = total_response[:idx + len('<|endofthink|>')]

        return total_response

    def stream_generate(self, prompt):

        total_response = ''
        responses = Generation.call(
            model=self.model, prompt=prompt, stream=True, **self.generate_cfg)

        for response in responses:
            if response.status_code == HTTPStatus.OK:
                new_response = response.output['text']
                frame_text = new_response[len(total_response):]
                yield frame_text

                idx = total_response.find('<|endofthink|>')
                if idx != -1:
                    break
                total_response = new_response
            else:
                print('Code: %d, status: %s, message: %s' %
                      (response.status_code, response.code, response.message))
