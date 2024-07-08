import os
import uuid

import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.tools.utils.oss import OssStorage
from modelscope_agent.tools.utils.output_wrapper import AudioWrapper
from modelscope_agent.utils.utils import get_api_key, get_upload_url

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('sambert_tts')
class SambertTtsTool(BaseTool):
    description = 'Sambert语音合成服务，将文本转成语音'
    name = 'sambert_tts'
    parameters: list = [{
        'name': 'text',
        'description': '需要转成语音的文本',
        'required': True,
        'type': 'string'
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        self.oss = None
        super().__init__(cfg)

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        try:
            token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        tts_text = params['text']
        if tts_text is None or len(tts_text) == 0 or tts_text == '':
            raise ValueError('tts input text is valid')
        os.makedirs(WORK_DIR, exist_ok=True)
        wav_name = str(uuid.uuid4())[0:6] + '_sambert_tts_audio.wav'
        wav_file = os.path.join(WORK_DIR, wav_name)
        response = SpeechSynthesizer.call(
            model='sambert-zhijia-v1',
            format='wav',
            text=tts_text,
            api_key=token)
        if response.get_audio_data() is not None:
            with open(wav_file, 'wb') as f:
                f.write(response.get_audio_data())
            if 'use_tool_api' in kwargs and kwargs['use_tool_api']:
                try:
                    wav_url = self._upload_to_oss(wav_name, wav_file)
                except Exception as e:
                    return (
                        f'Failed to save the audio file to oss with error: {e}, '
                        'please check the oss information')
                return str(AudioWrapper(wav_url, **kwargs))
        else:
            raise ValueError(
                f'call sambert tts failed, request id: {response.get_response()}'
            )
        return str(AudioWrapper(wav_file, **kwargs))

    def _upload_to_oss(self, file_name: str, file_path: str):
        if self.oss is None:
            self.oss = OssStorage()
        # this path is for modelscope only, please double-check
        oss_path = os.path.join('tmp', self.name, file_name)
        self.oss.upload(file_path, oss_path)
        url = self.oss.get_signed_url(oss_path)
        return url
