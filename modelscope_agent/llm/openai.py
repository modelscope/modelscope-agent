import os

import openai
from modelscope_agent.agent_types import AgentType

from .base import LLM
from .utils import CustomOutputWrapper

openai.api_key = os.getenv('OPENAI_API_KEY')


class OpenAi(LLM):
    name = 'openai'

    def __init__(self, cfg):
        super().__init__(cfg)

        self.model = self.cfg.get('model', 'gpt-3.5-turbo')
        self.model_id = self.model
        self.api_base = self.cfg.get('api_base', 'https://api.openai.com/v1')
        self.agent_type = self.cfg.get('agent_type', AgentType.DEFAULT)

    def generate(self,
                 llm_artifacts,
                 functions=[],
                 function_call='none',
                 **kwargs):
        if self.agent_type != AgentType.Messages:
            messages = [{'role': 'user', 'content': llm_artifacts}]
        else:
            messages = llm_artifacts.get(
                'messages', {
                    'role':
                    'user',
                    'content':
                    'No entry from user - please suggest something to enter'
                })

        # call openai function call api
        assert isinstance(functions, list)
        if len(functions) > 0 and self.agent_type == AgentType.Messages:
            function_call = 'auto'

        # covert to stream=True with stream updating
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                api_base=self.api_base,
                messages=messages,
                functions=functions,
                function_call=function_call,
                stream=False)
        except Exception as e:
            print(f'input: {messages}, original error: {str(e)}')
            raise e

        # only use index 0 in choice
        message = CustomOutputWrapper.handle_message_chat_completion(response)

        # truncate content
        content = message['content']

        if self.agent_type == AgentType.MS_AGENT:
            idx = content.find('<|endofthink|>')
            if idx != -1:
                content = content[:idx + len('<|endofthink|>')]
            return content
        elif self.agent_type == AgentType.Messages:
            return message
        else:
            return content
