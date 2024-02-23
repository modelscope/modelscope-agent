import os
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.llm.base import BaseChatModel, register_llm
from modelscope_agent.utils.retry import retry
from openai import OpenAI


@register_llm('openai')
class OpenAi(BaseChatModel):

    def __init__(self,
                 model: str,
                 model_server: str,
                 is_chat: bool = True,
                 is_function_call: Optional[bool] = None,
                 support_stream: Optional[bool] = None,
                 **kwargs):
        super().__init__(model, model_server)

        api_base = kwargs.get('api_base', 'https://api.openai.com/v1').strip()
        api_key = kwargs.get('api_key',
                             os.getenv('OPENAI_API_KEY',
                                       default='EMPTY')).strip()
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.is_function_call = is_function_call
        self.is_chat = is_chat
        self.support_stream = support_stream

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        response = self.client.chat.completions.create(
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            stream=False,
            **kwargs)
        # TODO: error handling
        return response.choices[0].message.content

    def support_function_calling(self):
        if self.is_function_call is None:
            return super().support_function_calling()
        else:
            return self.is_function_call

    def support_raw_prompt(self) -> bool:
        if self.is_chat is None:
            return super().support_raw_prompt()
        else:
            # if not chat, then prompt
            return not self.is_chat

    @retry(max_retries=3, delay_seconds=0.5)
    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Dict]] = None,
             stop: Optional[List[str]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Iterator[str]]:
        if isinstance(self.support_stream, bool):
            stream = self.support_stream
        if self.support_raw_prompt():
            return self.chat_with_raw_prompt(
                prompt=prompt, stream=stream, stop=stop, **kwargs)
        if not messages and prompt and isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        return super().chat(
            messages=messages, stop=stop, stream=stream, **kwargs)

    def _out_generator(self, response):
        for chunk in response:
            if hasattr(chunk.choices[0], 'text'):
                yield chunk.choices[0].text

    def chat_with_raw_prompt(self,
                             prompt: str,
                             stream: bool = True,
                             **kwargs) -> str:
        max_tokens = kwargs.get('max_tokens', 2000)
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=stream,
            max_tokens=max_tokens)

        # TODO: error handling
        if stream:
            return self._out_generator(response)
        else:
            return response.choices[0].text

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
