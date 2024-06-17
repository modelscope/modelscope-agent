import os
from typing import Dict, Iterator, List, Optional

from modelscope_agent.utils.logger import agent_logger as logger

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

    def __init__(self,
                 model: str,
                 model_server: str,
                 support_fn_call: bool = True,
                 **kwargs):

        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError(
                "The package 'zhipuai' is required for this module. Please install it using 'pip install zhipuai'."
            ) from e
        super().__init__(model, model_server, support_fn_call=support_fn_call)
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
        logger.info(
            f'====> stream messages: {messages}, functions: {functions}')
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
        logger.info(
            f'====> no stream messages: {messages}, functions: {functions}')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=functions,
            tool_choice=tool_choice,
        )
        message = response.choices[0].message
        output = message.content if not functions else [message.model_dump()]

        return output
