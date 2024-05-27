import os
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.llm.base import BaseChatModel, register_llm
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.retry import retry
import requests

# Temp code for openai proxy
@register_llm('openai_proxy')
class OpenAiProxy(BaseChatModel):

    def __init__(self,
                 model: str,
                 model_server: str,
                 is_chat: bool = True,
                 is_function_call: Optional[bool] = None,
                 support_stream: Optional[bool] = None,
                 **kwargs):
        super().__init__(model, model_server, is_function_call)

        self.api_base = kwargs.get('api_base', 'http://47.88.8.18:8088/api/ask').strip()
        self.api_key = kwargs.get('api_key',
                             os.getenv('OPENAI_API_KEY',
                                       default='EMPTY')).strip()
        self.data = {
            "model": model,
            "messages": [],
            "max_tokens": 2048,
            'temperature': 0.0,
            "seed": 1234,
            "stream_options": {
                "include_usage": True
            }
        }
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        stop = self._update_stop_word(stop)
        logger.info(
            f'call openai api, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: True, args: {str(kwargs)}')
        self.data['messages'] = messages
        self.data["stream"] = True
        response = requests.post(self.api_base, json=self.data, headers=self.headers)
        response = self.stat_last_call_token_info(response)
        # TODO: error handling
        for chunk in response:
            # sometimes delta.content is None by vllm, we should not yield None
            if len(chunk.choices) > 0 and hasattr(
                    chunk.choices[0].delta,
                    'content') and chunk.choices[0].delta.content:
                logger.info(
                    f'call openai api success, output: {chunk.choices[0].delta.content}'
                )
                yield chunk.choices[0].delta.content

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        stop = self._update_stop_word(stop)
        # logger.info(
            # f'call openai api, model: {self.model}, messages: {str(messages)}, '
            # f'stop: {str(stop)}, stream: False, args: {str(kwargs)}')
        self.data['messages'] = messages
        self.data["stream"] = False
        response = requests.post(self.api_base, json=self.data, headers=self.headers)
        # self.stat_last_call_token_info(response)
        # logger.info(
        #     f'call openai api success, output: {response.choices[0].message.content}'
        # )
        # TODO: error handling
        response = response.json()
        print(response['data']['response'])
        return response['data']['response']['choices'][0]['message']['content']

    def support_function_calling(self):
        if self.is_function_call is None:
            return super().support_function_calling()
        else:
            return self.is_function_call

    @retry(max_retries=3, delay_seconds=0.5)
    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Dict]] = None,
             stop: Optional[List[str]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Iterator[str]]:

        return super().chat(
            messages=messages, stop=stop, stream=stream, **kwargs)

    def _out_generator(self, response):
        for chunk in response:
            if hasattr(chunk.choices[0], 'text'):
                yield chunk.choices[0].text
