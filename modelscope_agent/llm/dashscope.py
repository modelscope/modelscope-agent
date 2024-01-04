import os
from http import HTTPStatus
from typing import Dict, Iterator, List, Optional

import dashscope

from .base import BaseChatModel, register_llm


def stream_output(response):
    last_len = 0
    delay_len = 5
    in_delay = False
    text = ''
    for trunk in response:
        if trunk.status_code == HTTPStatus.OK:
            text = trunk.output.choices[0].message.content
            if (len(text) - last_len) <= delay_len:
                in_delay = True
                continue
            else:
                in_delay = False
                real_text = text[:-delay_len]
                now_rsp = real_text[last_len:]
                yield now_rsp
                last_len = len(real_text)
        else:
            err = '\nError code: %s. Error message: %s' % (trunk.code,
                                                           trunk.message)
            if trunk.code == 'DataInspectionFailed':
                err += '\n错误码: 数据检查失败。错误信息: 输入数据可能包含不适当的内容。'
            text = ''
            yield f'{err}'
    # with open('debug.json', 'w', encoding='utf-8') as writer:
    #     writer.write(json.dumps(trunk, ensure_ascii=False))
    if text and (in_delay or (last_len != len(text))):
        yield text[last_len:]


@register_llm('dashscope')
class DashScopeLLM(BaseChatModel):
    """
    Universal LLM model interface on dashscope
    """

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)

        dashscope.api_key = kwargs.get(
            'api_key', os.getenv('DASHSCOPE_API_KEY', default='')).strip()
        assert dashscope.api_key, 'DASHSCOPE_API_KEY is required.'

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        stop = stop or []
        top_p = kwargs.get('top_p', 0.8)

        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            stop_words=[{
                'stop_str': word,
                'mode': 'exclude'
            } for word in stop],
            top_p=top_p,
            result_format='message',
            stream=True,
        )
        return stream_output(response)

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        stop = stop or []
        top_p = kwargs.get('top_p', 0.8)

        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            result_format='message',
            stream=False,
            stop_words=[{
                'stop_str': word,
                'mode': 'exclude'
            } for word in stop],
            top_p=top_p,
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        else:
            err = 'Error code: %s, error message: %s' % (
                response.code,
                response.message,
            )
            return err