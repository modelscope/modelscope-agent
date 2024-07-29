from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.callbacks import BaseCallback
from modelscope_agent.llm.utils.llm_templates import get_model_stop_words
from modelscope_agent.utils.retry import retry
from modelscope_agent.utils.tokenization_utils import count_tokens
from modelscope_agent.utils.utils import print_traceback

LLM_REGISTRY = {}


def register_llm(name):

    def decorator(cls):
        LLM_REGISTRY[name] = cls
        return cls

    return decorator


def enable_llm_callback(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        callbacks = kwargs.pop('callbacks', None)
        stream = kwargs.get('stream', True)
        if callbacks:
            callbacks.on_llm_start(*args, **kwargs)
        response = func(self, *args, **kwargs)
        if callbacks:
            if stream:
                response = enable_stream_callback(self.model, response,
                                                  callbacks)
            else:
                response = enable_no_stream_callback(self.model, response,
                                                     callbacks)
        return response

    return wrapper


def enable_stream_callback(model, rsp, callbacks):
    for s in rsp:
        if callbacks:
            callbacks.on_llm_new_token(model, s)
        yield s

    callbacks.on_llm_end(model, '', stream=True)


def enable_no_stream_callback(model, rsp, callbacks):
    callbacks.on_llm_end(model, rsp, stream=False)
    return rsp


class FnCallNotImplError(NotImplementedError):
    pass


class TextCompleteNotImplError(NotImplementedError):
    pass


class BaseChatModel(ABC):
    """
    The base class of llm.

    LLM subclasses need to inherit it. They must implement interfaces _chat_stream and _chat_no_stream,
    which correspond to streaming output and non-streaming output respectively.
    Optionally implement chat_with_functions and chat_with_raw_prompt for function calling and text completion.

    """

    def __init__(self,
                 model: str,
                 model_server: str,
                 support_fn_call: bool = None):
        self._support_fn_call: Optional[bool] = support_fn_call
        self.model = model
        self.model_server = model_server
        self.max_length = 6000

        self.last_call_usage_info = {}

    # It is okay to use the same code to handle the output
    # regardless of whether stream is True or False, as follows:
    # ```py
    # for chunk in chat_model.chat(..., stream=True/False):
    #   response += chunk
    #   yield response
    # ```

    @retry(max_retries=3, delay_seconds=0.5)
    @enable_llm_callback
    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Dict]] = None,
             stop: Optional[List[str]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Iterator[str]]:
        """
        chat interface

        Args:
            prompt: The inputted str query
            messages: The inputted messages, such as [{'role': 'user', 'content': 'hello'}]
            stop: The stop words list. The model will stop when outputted to them.
            stream: Requires streaming or non-streaming output

        Returns:
            (1) When str: Generated str response from llm in non-streaming
            (2) When Iterator[str]: Streaming output strings
        """
        if self.support_raw_prompt():
            if prompt and isinstance(prompt, str):
                messages = [{'role': 'user', 'content': prompt}]

        assert len(messages) > 0, 'messages list must not be empty'

        if stream:
            return self._chat_stream(messages, stop=stop, **kwargs)
        else:
            return self._chat_no_stream(messages, stop=stop, **kwargs)

    @retry(max_retries=3, delay_seconds=0.5)
    @enable_llm_callback
    def chat_with_functions(self,
                            messages: List[Dict],
                            functions: Optional[List[Dict]] = None,
                            stream: bool = True,
                            **kwargs) -> Dict:
        """
        Function call interface

        Args:
            messages: The inputted messages, such as [{'role': 'user', 'content': 'draw a picture'}]
            functions: The function list, such as:
                [{
                    'name': 'get_current_weather',
                    'description': 'Get the current weather in a given location.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type':
                                'string',
                                'description':
                                'The city and state, e.g. San Francisco, CA',
                            },
                            'unit': {
                                'type': 'string',
                                'enum': ['celsius', 'fahrenheit'],
                            },
                        },
                        'required': ['location'],
                    },
                }]

        Returns:
            generated response message
        """
        functions = [{
            'type': 'function',
            'function': item
        } for item in functions]
        if stream:
            return self._chat_stream(messages, functions, **kwargs)
        else:
            return self._chat_no_stream(messages, functions, **kwargs)

    def chat_with_raw_prompt(self,
                             prompt: str,
                             stop: Optional[List[str]] = None,
                             **kwargs) -> str:
        """
        The text completion interface.

        Args:
            prompt: Continuation of text
            stop: Stop words list. The model will stop when outputted to them.
        """
        raise TextCompleteNotImplError

    @abstractmethod
    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        """
        Streaming output interface.

        """
        raise NotImplementedError

    def _update_stop_word(self, stop=None):
        stop_words_from_model = get_model_stop_words(self.model)
        if stop is None:
            stop = stop_words_from_model
        else:
            stop = stop_words_from_model + stop

        return stop

    @abstractmethod
    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        """
        Non-streaming output interface.

        """
        raise NotImplementedError

    def support_function_calling(self) -> bool:
        """
        Check if LLM supports function calls
        """
        if self._support_fn_call is None:
            functions = [{
                'name': 'get_current_weather',
                'description': 'Get the current weather in a given location.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type':
                            'string',
                            'description':
                            'The city and state, e.g. San Francisco, CA',
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit'],
                        },
                    },
                    'required': ['location'],
                },
            }]
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
            except FnCallNotImplError:
                pass
            except AttributeError:
                pass
            except Exception:  # TODO: more specific
                print_traceback()
        return self._support_fn_call

    def support_raw_prompt(self) -> bool:
        """
        Check if LLM supports text completion.
        """
        try:
            self.chat_with_raw_prompt(prompt='')
            return True
        except TextCompleteNotImplError:
            return False
        except Exception:
            return False

    def check_max_length(self, messages: Union[List[Dict], str]) -> bool:
        if isinstance(messages, str):
            return count_tokens(messages) <= self.max_length
        total_length = 0
        for message in messages:
            total_length += count_tokens(message['content'])
        return total_length <= self.max_length

    def get_max_length(self) -> int:
        return self.max_length

    def get_usage(self) -> Dict:
        return self.last_call_usage_info

    def stat_last_call_token_info_stream(self, response):
        try:
            self.last_call_usage_info = response.usage.dict()
            return response
        except AttributeError:
            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    self.last_call_usage_info = chunk.usage.dict()
                yield chunk

    def stat_last_call_token_info_no_stream(self, response):
        if hasattr(response, 'usage'):
            self.last_call_usage_info = response.usage.dict()
        return response
