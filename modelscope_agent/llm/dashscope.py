import os
import re
from http import HTTPStatus
from typing import Dict, Iterator, List, Optional, Union

import dashscope
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.tokenization_utils import count_tokens

from .base import BaseChatModel, register_llm


def stream_output(response, **kwargs):
    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    last_len = 0
    delay_len = 5
    in_delay = False
    text = ''
    for trunk in response:
        if trunk.status_code == HTTPStatus.OK:
            # logging at the first frame for request_id, and the last frame for the whole output
            if not text:
                logger.info(
                    f'call dashscope generation api success, '
                    f'request_id: { trunk.request_id}, output: { trunk.output}'
                )
            try:
                text = trunk.output.choices[0].message.content
            except Exception:
                text = trunk.output.text
            text = text.split(im_end)[0].split(im_start)[0]
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
@register_llm('dashscope_llama3')
class DashScopeLLM(BaseChatModel):
    """
    Universal LLM model interface on dashscope
    """

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        self.max_length = kwargs.get(
            'max_length', int(os.getenv('DASHSCOPE_MAX_LENGTH', default=5650)))
        dashscope.api_key = kwargs.get(
            'api_key', os.getenv('DASHSCOPE_API_KEY', default='')).strip()
        assert dashscope.api_key, 'DASHSCOPE_API_KEY is required.'

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        stop = self._update_stop_word(stop)
        generation_input = {
            'model': self.model,
            'messages': messages,  # noqa
            'stop': [word for word in stop],
            'top_p': kwargs.get('top_p', 0.8),
            'result_format': 'message',
            'stream': True,
        }

        logger.query_info(
            uuid=kwargs.get('uuid_str', ''),
            details=generation_input,
            message='call dashscope generation api')
        if kwargs.get('temperature', None):
            generation_input['temperature'] = kwargs.get('temperature')
        if kwargs.get('seed', None):
            generation_input['seed'] = kwargs.get('seed')

        response = dashscope.Generation.call(**generation_input)
        response = self.stat_last_call_token_info_stream(response)
        return stream_output(response, **kwargs)

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        stop = self._update_stop_word(stop)
        top_p = kwargs.get('top_p', 0.8)

        response = dashscope.Generation.call(
            self.model,
            messages=messages,  # noqa
            result_format='message',
            stream=False,
            stop=[word for word in stop],
            top_p=top_p,
        )
        if response.status_code == HTTPStatus.OK:
            self.stat_last_call_token_info_no_stream(response)
            return response.output.choices[0].message.content
        else:
            err = 'Error code: %s, error message: %s' % (
                response.code,
                response.message,
            )
            return err

    def stat_last_call_token_info_no_stream(self, response):
        try:
            if response.usage is not None:
                if not response.usage.get('total_tokens'):
                    total_tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    total_tokens = response.usage.total_tokens
                self.last_call_usage_info = {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': total_tokens
                }
            else:
                logger.warning('No usage info in response')
        except AttributeError:
            logger.warning('No usage info in response')
        return response

    def stat_last_call_token_info_stream(self, response):
        try:
            if response.usage is not None:
                if not response.usage.get('total_tokens'):
                    total_tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    total_tokens = response.usage.total_tokens
                self.last_call_usage_info = {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': total_tokens
                }
            else:
                logger.warning('No usage info in response')
            return response
        except AttributeError:
            for chunk in response:
                try:
                    if not chunk.usage.get('total_tokens'):
                        total_tokens = chunk.usage.input_tokens + chunk.usage.output_tokens
                    else:
                        total_tokens = chunk.usage.total_tokens
                    self.last_call_usage_info = {
                        'prompt_tokens': chunk.usage.input_tokens,
                        'completion_tokens': chunk.usage.output_tokens,
                        'total_tokens': total_tokens
                    }
                except AttributeError:
                    logger.warning('No usage info in response')
                yield chunk


@register_llm('dashscope_qwen')
@register_llm('dashscope_qwen1.5')
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
        stop = self._update_stop_word(stop)
        top_p = kwargs.get('top_p', 0.8)

        response = dashscope.Generation.call(
            self.model,
            prompt=prompt,  # noqa
            stop=[word for word in stop],
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
            if message['role'] == 'user':
                query = message['content'].lstrip('\n').rstrip()
                local_prompt = f'\n{im_start}user\n{query}{im_end}'
            elif message['role'] == 'assistant':
                response = message['content'].lstrip('\n').rstrip()
                local_prompt = f'\n{im_start}assistant\n{response}{im_end}'

            if message['role'] != 'system':
                cur_content_length = count_tokens(local_prompt)
                if used_length + cur_content_length > self.max_length:
                    break
                used_length += cur_content_length
                prompt = local_prompt + prompt

        prompt = system_prompt + prompt

        # add one empty reply for the last round of assistant
        # ensure the end of prompt is assistant
        if not prompt.endswith(f'\n{im_start}assistant\n{im_end}'):
            prompt += f'\n{im_start}assistant\n{im_end}'
        prompt = prompt[:-len(f'{im_end}')]
        return prompt

    def build_multi_role_raw_prompt(self, messages: list):
        prompt = ''
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
        else:
            system_prompt = f'{im_start}system\nYou are a helpful assistant.{im_end}'

        # select user
        if 'recent_records' in system_prompt and 'chat_records' in system_prompt:
            chat_records = messages[-1]['content'].strip()
            recent_records = chat_records.split('\n')[-1]
            prompt = f'{system_prompt.replace("chat_records", chat_records).replace("recent_records", recent_records)}<|im_start|>assistant\n'  # noqa E501
        else:
            try:
                re_pattern_config = re.compile(pattern=r'你是([\s\S]+)，角色介绍')
                res = re_pattern_config.search(system_prompt)
                cur_role_name = res.group(1).strip()
            except Exception:
                cur_role_name = 'assistant'
            print('cur_role_name: ', cur_role_name)
            prompt = system_prompt
            content = messages[-1]['content'].lstrip('\n').rstrip()
            if 'chat_records' in prompt:
                prompt = f'{prompt.replace("chat_records", content)}\n<|im_start|>{cur_role_name}\n'
            else:
                chat_records_list = content.strip().split('\n')
                user_content = ''
                for chat_role in chat_records_list:
                    try:
                        cur_role, cur_chat = chat_role.split(':')
                    except Exception:
                        continue
                    user_content += f'<|im_start|>{cur_role.strip()}\n{cur_chat.strip()}<|im_end|>\n'
                prompt = f'{prompt}{user_content}<|im_start|>{cur_role_name}\n'

        return prompt

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        if self.model == 'qwen-spark-plus':
            return self._chat_stream_with_raw_prompt(messages, stop, **kwargs)
        else:
            return super()._chat_stream(messages, stop, **kwargs)

    def _chat_stream_with_raw_prompt(self,
                                     messages: List[Dict],
                                     stop: Optional[List[str]] = None,
                                     **kwargs) -> Iterator[str]:
        stop = self._update_stop_word(stop)
        generation_input = {
            'model': self.model,
            'prompt': messages[0]['content'],
            'stop': [word for word in stop],
            'top_p': kwargs.get('top_p', 0.95),
            'temperature': kwargs.get('temperature', 0.92),
            'result_format': 'message',
            'stream': True,
            'use_raw_prompt': True,
            'max_length': 100
        }

        logger.query_info(
            uuid=kwargs.get('uuid_str', ''),
            details=generation_input,
            message='call dashscope generation api')
        if kwargs.get('temperature', None):
            generation_input['temperature'] = kwargs.get('temperature')
        if kwargs.get('seed', None):
            generation_input['seed'] = kwargs.get('seed')
        response = dashscope.Generation.call(**generation_input)
        return stream_output(response, **kwargs)
