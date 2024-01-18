import os
from typing import Dict, Iterator, List, Optional

from zhipuai import ZhipuAI

from .base import BaseChatModel, register_llm


def stream_output(response, **kwargs):
    func_call = {
        'name': None,
        'arguments': '',
    }
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            # TODO : multi tool_calls
            tool_call = delta.tool_calls[0]
            print(f'tool_call: {tool_call}')
            func_call['name'] = tool_call.function.name
            func_call['arguments'] += tool_call.function.arguments
        if chunk.choices[0].finish_reason == 'tool_calls':
            yield {'function_call': func_call}
        else:
            yield delta.content


@register_llm('zhipu')
class ZhipuLLM(BaseChatModel):
    """
    Universal LLM model interface on zhipu
    """

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        self._support_fn_call = True
        api_key = kwargs.get('api_key', os.getenv('ZHIPU_API_KEY', '')).strip()
        assert api_key, 'ZHIPU_API_KEY is required.'
        self.client = ZhipuAI(api_key=api_key)

    def _chat_stream(self,
                     messages: List[Dict],
                     functions: Optional[List[Dict]] = None,
                     tool_choice='auto',
                     **kwargs) -> Iterator[str]:
        if not functions or not len(functions):
            tool_choice = 'none'
        print(f'====> stream messages: {messages}')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=functions,
            tool_choice=tool_choice,
            stream=True,
        )
        return stream_output(response, **kwargs)

    def _chat_no_stream(self,
                        messages: List[Dict],
                        functions: Optional[List[Dict]] = None,
                        tool_choice='auto',
                        **kwargs) -> str:
        if not functions or not len(functions):
            tool_choice = 'none'
        print(f'====> no stream messages: {messages}')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=functions,
            tool_choice=tool_choice,
        )
        return response.choices[0].message

    def chat_with_functions(self,
                            messages: List[Dict],
                            functions: Optional[List[Dict]] = None,
                            stream: bool = True,
                            **kwargs) -> Dict:
        functions = [{
            'type': 'function',
            'function': item
        } for item in functions]
        if stream:
            return self._chat_stream(messages, functions, **kwargs)
        else:
            return self._chat_no_stream(messages, functions, **kwargs)


@register_llm('glm-4')
class GLM4(ZhipuLLM):
    """
    glm-4 from zhipu
    """
