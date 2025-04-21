import os
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.llm.base import BaseChatModel, register_llm
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.retry import retry
from modelscope_agent.utils.utils import print_traceback

from openai import AzureOpenAI, OpenAI


@register_llm('openai')
@register_llm('azure_openai')
class OpenAi(BaseChatModel):

    def __init__(
        self,
        model: str,
        model_server: str,
        is_chat: bool = True,
        is_function_call: Optional[bool] = None,
        support_stream: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(model, model_server, is_function_call)

        self.is_azure = model_server.lower().startswith('azure')
        if self.is_azure:
            default_azure_endpoint = os.getenv(
                'AZURE_OPENAI_ENDPOINT',
                'https://docs-test-001.openai.azure.com/')
            azure_endpoint = kwargs.get('azure_endpoint',
                                        default_azure_endpoint).strip()
            api_key = kwargs.get(
                'api_key', os.getenv('AZURE_OPENAI_API_KEY',
                                     default='EMPTY')).strip()
            api_version = kwargs.get('api_version', '2024-06-01').strip()
            logger.info(
                f'client url {azure_endpoint}, client key: {api_key}, client version: {api_version}'
            )

            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            default_api_base = os.getenv('OPENAI_API_BASE',
                                         'https://api.openai.com/v1')
            api_base = kwargs.get('api_base', default_api_base).strip()
            api_key = kwargs.get('api_key',
                                 os.getenv('OPENAI_API_KEY',
                                           default='EMPTY')).strip()
            logger.info(f'client url {api_base}, client key: {api_key}')

            self.client = OpenAI(api_key=api_key, base_url=api_base)

        self.is_function_call = is_function_call
        self.is_chat = is_chat
        self.support_stream = support_stream

    def support_function_calling(self) -> bool:
        """
        Check if LLM supports function calls
        """
        if self._support_fn_call is None:
            functions = [{'name': 'fetch.fetch', 'description': 'Fetches a URL from the internet and optionally extracts its contents as markdown.\n\nAlthough originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.', 'input_schema': {'description': 'Parameters for fetching a URL.', 'properties': {'url': {'description': 'URL to fetch', 'format': 'uri', 'minLength': 1, 'title': 'Url', 'type': 'string'}, 'max_length': {'default': 5000, 'description': 'Maximum number of characters to return.', 'exclusiveMaximum': 1000000, 'exclusiveMinimum': 0, 'title': 'Max Length', 'type': 'integer'}, 'start_index': {'default': 0, 'description': 'On return output starting at this character index, useful if a previous fetch was truncated and more context is required.', 'minimum': 0, 'title': 'Start Index', 'type': 'integer'}, 'raw': {'default': False, 'description': 'Get the actual HTML content of the requested page, without simplification.', 'title': 'Raw', 'type': 'boolean'}}, 'required': ['url'], 'title': 'Fetch', 'type': 'object'}}]
            messages = [{
                'role': 'user',
                'content': 'What is the weather like in Boston?'
            }]
            self._support_fn_call = False
            try:
                response = self.chat_with_functions(
                    messages=messages, functions=functions)
                if response.get('function_call', None):
                    # logger.info('Support of function calling is detected.')
                    self._support_fn_call = True
                if response.get('tool_calls', None):
                    # logger.info('Support of function calling is detected.')
                    self._support_fn_call = True

            except AttributeError:
                pass
            except Exception:  # TODO: more specific
                print_traceback()
        return self._support_fn_call

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        stop = self._update_stop_word(stop)
        logger.info(
            f'call openai api, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: True, args: {str(kwargs)}')

        if not self.is_azure:
            kwargs['stream_options'] = {'include_usage': True}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            stream=True,
            **kwargs)

        response = self.stat_last_call_token_info_stream(response)
        # TODO: error handling
        for chunk in response:
            # sometimes delta.content is None by vllm, we should not yield None
            if (len(chunk.choices) > 0
                    and hasattr(chunk.choices[0].delta, 'content')
                    and chunk.choices[0].delta.content):
                logger.info(
                    f'call openai api success, output: {chunk.choices[0].delta.content}'
                )
                yield chunk.choices[0].delta.content

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        stop = self._update_stop_word(stop)
        logger.info(
            f'call openai api, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: False, args: {str(kwargs)}')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            stream=False,
            **kwargs)
        self.stat_last_call_token_info_no_stream(response)
        logger.info(
            f'call openai api success, output: {response.choices[0].message.content}'
        )
        # TODO: error handling
        return response.choices[0].message.content

    def support_raw_prompt(self) -> bool:
        if self.is_chat is None:
            return super().support_raw_prompt()
        else:
            # if not chat, then prompt
            return not self.is_chat

    @retry(max_retries=3, delay_seconds=0.5)
    def chat(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Iterator[str]]:

        if 'uuid_str' in kwargs:
            kwargs.pop('uuid_str')
        if 'append_files' in kwargs:
            kwargs.pop('append_files')
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
        print(f'functions: {functions}')
        if functions:
            functions = [{
                'type': 'function',
                'function': item
            } for item in functions]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=functions,
                tool_choice='auto',
                **kwargs,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs)
        # TODO: error handling
        return response.choices[0].message


@register_llm('vllm-server')
class Vllm(OpenAi):

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        stop = self._update_stop_word(stop)
        logger.info(
            f'call openai api, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: True, args: {str(kwargs)}')
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stop=stop, stream=True)
        response = self.stat_last_call_token_info_stream(response)
        # TODO: error handling
        for chunk in response:
            # sometimes delta.content is None by vllm, we should not yield None
            if (len(chunk.choices) > 0
                    and hasattr(chunk.choices[0].delta, 'content')
                    and chunk.choices[0].delta.content):
                logger.info(
                    f'call openai api success, output: {chunk.choices[0].delta.content}'
                )
                yield chunk.choices[0].delta.content

