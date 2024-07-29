from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.retry import retry

from .base import BaseChatModel, register_llm


@register_llm('ollama')
class OllamaLLM(BaseChatModel):

    def __init__(self, model: str, model_server: str, **kwargs):
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "The package 'ollama' is required for this module. Please install it using 'pip install ollama'."
            ) from e
        super().__init__(model, model_server)
        host = kwargs.get('host', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
        self.model = model
        try:
            logger.debug(f'Pulling model {self.model}')
            self.client.pull(self.model)
        except Exception as e:
            logger.warning(
                f'Warning: Failed to pull model {self.model} from {host}: {e}')

        logger.info(
            f'Initialization of OllamaLLM with model {self.model} done')

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        logger.info(
            f'call ollama, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: True, args: {str(kwargs)}')
        stream = self.client.chat(
            model=self.model, messages=messages, stream=True)
        stream = self.stat_last_call_token_info_stream(stream)
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
        self.stat_last_call_token_info_no_stream(response)
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

    def stat_last_call_token_info_no_stream(self, response):
        try:
            self.last_call_usage_info = {
                'prompt_tokens':
                response.get('prompt_eval_count', -1),
                'completion_tokens':
                response.get('eval_count', -1),
                'total_tokens':
                response.get('prompt_eval_count') + response.get('eval_count')
            }
        except AttributeError:
            logger.warning('No usage info in response')

        return response

    def stat_last_call_token_info_stream(self, response):
        try:
            self.last_call_usage_info = {
                'prompt_tokens':
                response.get('prompt_eval_count', -1),
                'completion_tokens':
                response.get('eval_count', -1),
                'total_tokens':
                response.get('prompt_eval_count') + response.get('eval_count')
            }
            return response
        except AttributeError:
            for chunk in response:
                # if hasattr(chunk.output, 'usage'):
                self.last_call_usage_info = {
                    'prompt_tokens':
                    chunk.get('prompt_eval_count', -1),
                    'completion_tokens':
                    chunk.get('eval_count', -1),
                    'total_tokens':
                    chunk.get('prompt_eval_count', -1)
                    + chunk.get('eval_count', -1)
                }
                yield chunk
