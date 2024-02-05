import os
from http import HTTPStatus
from typing import Dict, Iterator, List, Optional

import dashscope
from modelscope_agent.utils.logger import agent_logger as logger

from .base import BaseChatModel, register_llm


def stream_output(response, **kwargs):
    last_len = 0
    delay_len = 5
    in_delay = False
    text = ''
    for trunk in response:
        if trunk.status_code == HTTPStatus.OK:
            # logger at the first for the request_id, and the last time for whole output
            if not text or trunk.output.choices[0].finish_reason != 'null':
                logger.info(
                    f'call dashscope generation api success, '
                    f'request_id: { trunk.request_id}, output: { trunk.output}'
                )
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
                err += '\n错误码: 数据检查失败。错误信息: 输入数据可能包含不适当的内容。由于该不适当内容会一直存在历史对话中，后续的对话大概率仍会触发此错误。建议刷新重置页面。'
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
        generation_input = {
            'model': self.model,
            'messages': messages,  # noqa
            'stop_words': [{
                'stop_str': word,
                'mode': 'exclude'
            } for word in stop],
            'top_p': kwargs.get('top_p', 0.8),
            'result_format': 'message',
            'stream': True,
        }
        response = dashscope.Generation.call(**generation_input)
        return stream_output(response, **kwargs)

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


@register_llm('dashscope_qwen')
@register_llm('dashscope_qwen2')
class QwenChatAtDS(DashScopeLLM):
    """
    qwen_model from dashscope
    """

    def chat_with_raw_prompt(self,
                             prompt: str,
                             stop: Optional[List[str]] = None,
                             **kwargs) -> str:
        if prompt == '':
            return ''
        stop = stop or []
        top_p = kwargs.get('top_p', 0.8)

        response = dashscope.Generation.call(
            self.model,
            prompt=prompt,  # noqa
            stop_words=[{
                'stop_str': word,
                'mode': 'exclude'
            } for word in stop],
            top_p=top_p,
            result_format='message',
            stream=False,
            use_raw_prompt=True,
        )
        if response.status_code == HTTPStatus.OK:
            # with open('debug.json', 'w', encoding='utf-8') as writer:
            #     writer.write(json.dumps(response, ensure_ascii=False))
            return response.output.choices[0].message.content
        else:
            err = 'Error code: %s, error message: %s' % (
                response.code,
                response.message,
            )
            return err

    def build_raw_prompt(self, messages):
        messages.append({'role': 'assistant', 'content': ''})
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0]['role'] == 'system':
            sys = messages[0]['content']
            prompt = f'{im_start}system\n{sys}{im_end}'
        else:
            prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'

        for message in messages:
            if message['role'] == 'user':
                query = message['content'].lstrip('\n').rstrip()
                prompt += f'\n{im_start}user\n{query}{im_end}'
            elif message['role'] == 'assistant':
                response = message['content'].lstrip('\n').rstrip()
                prompt += f'\n{im_start}assistant\n{response}{im_end}'

        # add one empty reply for the last round of assistant
        assert prompt.endswith(f'\n{im_start}assistant\n{im_end}')
        prompt = prompt[:-len(f'{im_end}')]
        return prompt
