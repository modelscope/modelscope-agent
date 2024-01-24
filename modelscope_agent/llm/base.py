from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.utils.utils import print_traceback

LLM_REGISTRY = {}


def register_llm(name):

    def decorator(cls):
        LLM_REGISTRY[name] = cls
        return cls

    return decorator


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

    def __init__(self, model: str, model_server: str):
        self._support_fn_call: Optional[bool] = None
        self.model = model
        self.model_server = model_server

    # It is okay to use the same code to handle the output
    # regardless of whether stream is True or False, as follows:
    # ```py
    # for chunk in chat_model.chat(..., stream=True/False):
    #   response += chunk
    #   yield response
    # ```
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

    def chat_with_functions(self,
                            messages: List[Dict],
                            functions: Optional[List[Dict]] = None,
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
        raise FnCallNotImplError

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
            except FnCallNotImplError:
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
