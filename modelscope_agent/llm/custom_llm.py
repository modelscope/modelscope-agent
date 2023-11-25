import os

import json
import requests
from modelscope_agent.agent_types import AgentType

from .base import LLM
from .utils import DEFAULT_MESSAGE


class CustomLLM(LLM):
    '''
        This method is for the service that provide llm serving through http.
        user could override the result parsing method if needed
        While put all the necessary information in the env variable, such as Token, Model, URL
    '''
    name = 'custom_llm'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.token = os.getenv('HTTP_LLM_TOKEN', None)
        self.model = os.getenv('HTTP_LLM_MODEL', None)
        self.model_id = self.model
        self.url = os.getenv('HTTP_LLM_URL', None)

        if self.token is None:
            raise ValueError('HTTP_LLM_TOKEN is not set')
        self.agent_type = self.cfg.get('agent_type', AgentType.DEFAULT)

    def http_request(self, data):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        response = requests.post(self.url, json=data, headers=headers)
        return json.loads(response.content)

    def generate(self,
                 llm_artifacts,
                 functions=[],
                 function_call='none',
                 **kwargs):
        if self.agent_type != AgentType.Messages:
            messages = [{'role': 'user', 'content': llm_artifacts}]
        else:
            messages = llm_artifacts if len(
                llm_artifacts) > 0 else DEFAULT_MESSAGE

        data = {'model': self.model, 'messages': messages, 'n': 1}

        assert isinstance(functions, list)
        if len(functions) > 0:
            function_call = 'auto'
            data['functions'] = functions
            data['function_call'] = function_call

        retry_count = 0
        max_retries = 3
        message = {'content': ''}
        while retry_count <= max_retries:

            try:
                response = self.http_request(data)
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f'input: {messages}, original error: {str(e)}')
                    raise e

            if response['code'] == 200:
                message = response['data']['response'][0]
                break
            else:
                retry_count += 1
                if retry_count > max_retries:
                    print('maximum retry reached, return default message')

        # truncate content
        content = message['content']

        if self.agent_type == AgentType.MS_AGENT:
            idx = content.find('<|endofthink|>')
            if idx != -1:
                content = content[:idx + len('<|endofthink|>')]
            return content
        elif self.agent_type == AgentType.Messages:
            new_message = {
                'content': content,
                'role': message.get('response_role', 'assistant')
            }
            if 'function_call' in message and message['function_call'] != {}:
                new_message['function_call'] = message.get('function_call')
            return new_message
        else:
            return content
