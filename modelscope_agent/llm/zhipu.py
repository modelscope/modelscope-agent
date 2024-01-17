import os
from typing import Dict, Iterator, List, Optional

from zhipuai import ZhipuAI

from .base import BaseChatModel, register_llm


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


def stream_output(response, **kwargs):
    func_call = {
        'name': None,
        'arguments': '',
    }
    for chunk in response:
        delta = chunk.choices[0].delta
        if 'tool_calls' in delta:
            if 'name' in delta.function_call:
                func_call['name'] = delta.function_call['name']
            if 'arguments' in delta.function_call:
                func_call['arguments'] += delta.function_call['arguments']
        if chunk.choices[0].finish_reason == 'tool_calls':
            yield func_call
        else:
            yield delta.content


@register_llm('glm-4')
class GLM4(ZhipuLLM):
    """
    qwen_model from dashscope
    """

    def chat_with_functions_stream(self,
                                   messages: List[Dict],
                                   functions: Optional[List[Dict]] = None,
                                   tool_choice='none',
                                   **kwargs) -> Iterator[str]:
        if len(functions) > 0:
            tool_choice = 'auto'
        response = self.client.chat.asyncCompletions.create(
            model='glm-4',
            messages=messages,
            tools=functions,
            tool_choice=tool_choice,
        )
        return stream_output(response, **kwargs)
