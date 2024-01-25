import os

from modelscope_agent.output_wrapper import AudioWrapper
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError
from dashscope.audio.tts import SpeechSynthesizer

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


class SambertTtsTool(Tool):
    description = 'Sambert语音合成服务，将文本转成语音'
    name = 'sambert_tts'
    parameters: list = [{
        'name': 'text',
        'description': '需要转成语音的文本',
        'required': True
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})

        self.api_key = self.cfg.get('dashscope_api_key', os.environ.get('DASHSCOPE_API_KEY'))
        if self.api_key is None:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)

    def __call__(self, *args, **kwargs):
        tts_text = kwargs['text']
        if tts_text is None or len(tts_text) == 0 or tts_text == '':
            raise ValueError(f'tts input text is valid')
        os.makedirs(WORK_DIR, exist_ok=True)
        wav_file = WORK_DIR + '/sambert_tts_audio.wav'
        response = SpeechSynthesizer.call(model='sambert-zhijia-v1', format='wav', text=tts_text)
        if response.get_audio_data() is not None:
            with open(wav_file, 'wb') as f:
                f.write(response.get_audio_data())
        else:
            raise ValueError(f'call sambert asr failed, request id: {response.get_response().request_id}')
        return {'result': AudioWrapper(wav_file)}


if __name__ == '__main__':
    tool = SambertTtsTool()
    tool(text='今天天气怎么样？')
