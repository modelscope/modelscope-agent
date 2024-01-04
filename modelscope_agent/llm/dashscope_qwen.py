from http import HTTPStatus
from typing import List, Optional

import dashscope
from modelscope_agent.llm.base import register_llm
from modelscope_agent.llm.dashscope import DashScopeLLM


@register_llm('dashscope_qwen')
class QwenChatAtDS(DashScopeLLM):
    """
    qwen_model from dashscope
    """

    def chat_with_raw_prompt(self,
                             prompt: str,
                             stop: Optional[List[str]] = None,
                             **kwargs) -> str:
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
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0]['role'] == 'system':
            sys = messages[0]['role']
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
