from zhipuai import ZhipuAI
from .base import BaseChatModel, register_llm


class ZhipuLLM(BaseChatModel):
    """
    Universal LLM model interface on zhipu
    """

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        api_key = kwargs.get('api_key', os.getenv('ZHIPU_API_KEY', '')).strip()
        assert api_key, 'ZHIPU_API_KEY is required.'
        self.client = ZhipuAI(api_key=api_key)


def stream_output(response, **kwargs):
    for chunk in response:
        yield chunk.choices[0].delta


@register_llm('glm-4')
class GLM4(ZhipuLLM):
    """
    qwen_model from dashscope
    """
    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        stop = stop or []
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=messages,
            stream=True,
        )
        return stream_output(response, **kwargs)

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
