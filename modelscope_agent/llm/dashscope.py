import os
from http import HTTPStatus
from typing import Dict, Iterator, List, Optional, Union

import dashscope
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.tokenization_utils import count_tokens

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
            logger.query_error(
                uuid=kwargs.get('uuid_str', ''),
                details={
                    'dashscope.request_id': trunk.request_id,
                    'dashscope.status_code': trunk.status_code,
                    'dashscope.code': trunk.code,
                    'dashscope.message': trunk.message
                },
                message='call dashscope generation api error')

            err = '\nError code: %s. Error message: %s with request id %s' % (
                trunk.code, trunk.message, trunk.request_id)
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
        self.max_length = kwargs.get(
            'max_length', int(os.getenv('DASHSCOPE_MAX_LENGTH', default=6000)))
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

        logger.query_info(
            uuid=kwargs.get('uuid_str', ''),
            details=generation_input,
            message='call dashscope generation api')
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

    def build_raw_prompt(self, messages: list):
        prompt = ''
        messages.append({'role': 'assistant', 'content': ''})
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0]['role'] == 'system':
            sys = messages[0]['content']
            system_prompt = f'{im_start}system\n{sys}{im_end}'
        else:
            system_prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'

        used_length = count_tokens(system_prompt)

        for message in reversed(messages):
            cur_content_length = count_tokens(message['content'])
            if used_length + cur_content_length > self.max_length:
                break
            used_length += cur_content_length
            if message['role'] == 'user':
                query = message['content'].lstrip('\n').rstrip()
                prompt = f'\n{im_start}user\n{query}{im_end}' + prompt
            elif message['role'] == 'assistant':
                response = message['content'].lstrip('\n').rstrip()
                prompt = f'\n{im_start}assistant\n{response}{im_end}' + prompt

        prompt = system_prompt + prompt

        # add one empty reply for the last round of assistant
        # ensure the end of prompt is assistant
        if not prompt.endswith(f'\n{im_start}assistant\n{im_end}'):
            prompt += f'\n{im_start}assistant\n{im_end}'
        prompt = prompt[:-len(f'{im_end}')]
        return prompt
