import os
import random
from http import HTTPStatus
from typing import Union

import dashscope
from dashscope import Generation
from modelscope_agent.agent_types import AgentType

from .base import LLM
from .utils import DEFAULT_MESSAGE

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')


class DashScopeLLM(LLM):
    name = 'dashscope_llm'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.cfg.get('model', 'modelscope-agent-llm-v1')
        self.generate_cfg = self.cfg.get('generate_cfg', {})
        self.agent_type = self.cfg.get('agent_type', AgentType.DEFAULT)

    def set_agent_type(self, agent_type):
        self.agent_type = agent_type

    def generate(self,
                 llm_artifacts: Union[str, dict],
                 functions=[],
                 **kwargs):
        total_response = ''
        # TODO retry and handle message
        if self.agent_type == AgentType.OPENAI_FUNCTIONS:
            messages = llm_artifacts if len(
                llm_artifacts) > 0 else DEFAULT_MESSAGE
            responses = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                # set the random seed, optional, default to 1234 if not set
                seed=random.randint(1, 10000),
                result_format=
                'message',  # set the result to be "message" format.
                stream=False,
                **self.generate_cfg)
        else:
            responses = Generation.call(
                model=self.model,
                prompt=llm_artifacts,
                stream=False,
                **self.generate_cfg)

        if responses.status_code == HTTPStatus.OK:
            total_response = responses.output['text']
        else:
            print('Code: %d, status: %s, message: %s' %
                  (responses.status_code, responses.code, responses.message))

        idx = total_response.find('<|endofthink|>')
        if idx != -1:
            total_response = total_response[:idx + len('<|endofthink|>')]

        return total_response

    def stream_generate(self, prompt, **kwargs):

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
