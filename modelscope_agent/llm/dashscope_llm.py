import os
import random
import traceback
from http import HTTPStatus
from typing import Union

import dashscope
import json
from dashscope import Generation
from modelscope_agent.agent_types import AgentType

from .base import LLM
from .utils import DEFAULT_MESSAGE, CustomOutputWrapper

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')


class DashScopeLLM(LLM):
    name = 'dashscope_llm'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.cfg.get('model', 'modelscope-agent-llm-v1')
        self.generate_cfg = self.cfg.get('generate_cfg', {})
        self.agent_type = self.cfg.get('agent_type', AgentType.DEFAULT)

    def generate(self,
                 llm_artifacts: Union[str, dict],
                 functions=[],
                 **kwargs):

        # TODO retry and handle message
        try:
            if self.agent_type == AgentType.Messages:
                messages = llm_artifacts if len(
                    llm_artifacts) > 0 else DEFAULT_MESSAGE
                self.generate_cfg['use_raw_prompt'] = False
                response = dashscope.Generation.call(
                    model=self.model,
                    messages=messages,
                    # set the random seed, optional, default to 1234 if not set
                    seed=random.randint(1, 10000),
                    result_format=
                    'message',  # set the result to be "message" format.
                    stream=False,
                    **self.generate_cfg)
                llm_result = CustomOutputWrapper.handle_message_chat_completion(
                    response)
            else:
                response = Generation.call(
                    model=self.model,
                    prompt=llm_artifacts,
                    stream=False,
                    **self.generate_cfg)
                llm_result = CustomOutputWrapper.handle_message_text_completion(
                    response)
            return llm_result
        except Exception as e:
            error = traceback.format_exc()
            print(
                f'LLM error with input {llm_artifacts}, original error: {str(e)} with detail {error}'
            )
            raise RuntimeError(error)

        if self.agent_type == AgentType.MS_AGENT:
            # in the form of text
            idx = llm_result.find('<|endofthink|>')
            if idx != -1:
                llm_result = llm_result[:idx + len('<|endofthink|>')]
            return llm_result
        elif self.agent_type == AgentType.Messages:
            # in the form of message
            return llm_result
        else:
            # in the form of text
            return llm_result

    #
    # def stream_generate(self, prompt, functions, **kwargs):
    #
    #     total_response = ''
    #     responses = Generation.call(
    #         model=self.model, prompt=prompt, stream=True, **self.generate_cfg)
    #
    #     for response in responses:
    #         if response.status_code == HTTPStatus.OK:
    #             new_response = response.output['text']
    #             frame_text = new_response[len(total_response):]
    #             yield frame_text
    #
    #             idx = total_response.find('<|endofthink|>')
    #             if idx != -1:
    #                 break
    #             total_response = new_response
    #         else:
    #             print('Code: %d, status: %s, message: %s' %
    #                   (response.status_code, response.code, response.message))
