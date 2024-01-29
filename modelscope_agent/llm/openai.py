import os
from typing import Dict, Iterator, List, Optional

from openai import OpenAI
from modelscope_agent.llm.base import BaseChatModel, register_llm


@register_llm('openai')
class OpenAi(BaseChatModel):

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)

        api_base = kwargs.get('api_base',
                                     'https://api.openai.com/v1').strip()
        api_key = kwargs.get(
            'api_key', os.getenv('OPENAI_API_KEY', default='EMPTY')).strip()
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        response = self.client.completions.create(
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
        response = self.client.completions.create(
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
            response = self.client.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                **kwargs)
        else:
            response = self.client.completions.create(
                model=self.model, messages=messages, **kwargs)
        # TODO: error handling
        return response.choices[0].message


@register_llm('openapi')
class OpenAPILocal(BaseChatModel):
    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )       

    def _chat_stream(self,
                     prompt: str,
                     **kwargs) -> Iterator[str]:
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            )
        # TODO: error handling
        for chunk in response:
            if hasattr(chunk.choices[0], 'text'):
                yield chunk.choices[0].text

    def _chat_no_stream(self,
                        prompt: str,
                        **kwargs) -> str:
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=False,
            )
        # TODO: error handling
        return response.choices[0].message.content

    def support_function_calling(self) -> bool:
        return False
