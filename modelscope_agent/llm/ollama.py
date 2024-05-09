from typing import Dict, Iterator, List, Optional, Union

import ollama
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.retry import retry

from .base import BaseChatModel, register_llm


@register_llm('ollama')
class OllamaLLM(BaseChatModel):

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        host = kwargs.get('host', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
        self.model = model

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        logger.info(
            f'call ollama, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: True, args: {str(kwargs)}')
        stream = self.client.chat(
            model=self.model, messages=messages, stream=True)
        for chunk in stream:
            tmp_content = chunk['message']['content']
            logger.info(f'call ollama success, output: {tmp_content}')
            if stop and any(word in tmp_content for word in stop):
                break
            yield tmp_content

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        logger.info(
            f'call ollama, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: False, args: {str(kwargs)}')
        response = self.client.chat(model=self.model, messages=messages)
        final_content = response['message']['content']
        logger.info(f'call ollama success, output: {final_content}')
        return final_content

    def support_raw_prompt(self) -> bool:
        return super().support_raw_prompt()

    def _out_generator(self, response):
        for chunk in response:
            if hasattr(chunk['message']['content'], 'text'):
                yield chunk['message']['content'].text

    def chat_with_raw_prompt(self,
                             prompt: str,
                             stream: bool = True,
                             **kwargs) -> str:
        max_tokens = kwargs.get('max_tokens', 2000)
        response = self.client.chat(
            model=self.model,
            prompt=prompt,
            stream=stream,
            max_tokens=max_tokens)

        # TODO: error handling
        if stream:
            return self._out_generator(response)
        else:
            return response['message']['content']

    @retry(max_retries=3, delay_seconds=0.5)
    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Dict]] = None,
             stop: Optional[List[str]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Iterator[str]]:
        if self.support_raw_prompt():
            return self.chat_with_raw_prompt(
                prompt=prompt, stream=stream, stop=stop, **kwargs)
        if not messages and prompt and isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        return super().chat(
            messages=messages, stop=stop, stream=stream, **kwargs)
