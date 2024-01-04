import os
from typing import Dict, Iterator, List, Optional

import openai
from agent_scope.llm.base import BaseChatModel, register_llm


@register_llm('openai')
class OpenAi(BaseChatModel):

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)

        openai.api_base = kwargs.get('api_base',
                                     'https://api.openai.com/v1').strip()
        openai.api_key = kwargs.get(
            'api_key', os.getenv('OPENAI_API_KEY', default='EMPTY')).strip()

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stop=stop,
            stream=True,
            **kwargs)
        # TODO: error handling
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content'):
                yield chunk.choices[0].delta.content

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stop=stop,
            stream=False,
            **kwargs)
        # TODO: error handling
        return response.choices[0].message.content

    def chat_with_functions(self,
                            messages: List[Dict],
                            functions: Optional[List[Dict]] = None,
                            **kwargs) -> Dict:
        if functions:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                functions=functions,
                **kwargs)
        else:
            response = openai.ChatCompletion.create(
                model=self.model, messages=messages, **kwargs)
        # TODO: error handling
        return response.choices[0].message
