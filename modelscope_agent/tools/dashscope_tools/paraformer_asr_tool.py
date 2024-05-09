import os
import subprocess
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import get_api_key

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


def _preprocess(input_file, output_file):
    ret = subprocess.call([
        'ffmpeg', '-y', '-i', input_file, '-f', 's16le', '-acodec',
        'pcm_s16le', '-ac', '1', '-ar', '16000', '-loglevel', 'quiet',
        output_file
    ])
    if ret != 0:
        raise ValueError(f'Failed to preprocess audio file {input_file}')


@register_tool('paraformer_asr')
class ParaformerAsrTool(BaseTool):
    description = 'Paraformer语音识别服务，将语音转成文本'
    name = 'paraformer_asr'
    parameters: list = [{
        'name': 'audio_path',
        'description': '需要转成文本的语音文件路径',
        'required': True,
        'type': 'string'
    }]

    def __init__(self, cfg: Optional[Dict] = {}):
        self.cfg = cfg.get(self.name, {})

        self.api_key = self.cfg.get('dashscope_api_key',
                                    os.environ.get('DASHSCOPE_API_KEY', ''))
        super().__init__(cfg)

    def call(self, params: str, **kwargs):
        from dashscope.audio.asr import Recognition
        params = self._verify_args(params)
        try:
            self.api_key = get_api_key(ApiNames.dashscope_api_key,
                                       self.api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        raw_audio_file = WORK_DIR + '/' + params['audio_path']
        if not os.path.exists(raw_audio_file):
            raise ValueError(f'audio file {raw_audio_file} not exists')
        pcm_file = WORK_DIR + '/' + 'audio.pcm'
        _preprocess(raw_audio_file, pcm_file)
        if not os.path.exists(pcm_file):
            raise ValueError(f'convert audio to pcm file {pcm_file} failed')
        recognition = Recognition(
            model='paraformer-realtime-v1',
            format='pcm',
            sample_rate=16000,
            callback=None)
        response = recognition.call(pcm_file)
        result = ''
        if response.status_code == HTTPStatus.OK:
            sentences: List[Any] = response.get_sentence()
            if sentences and len(sentences) > 0:
                for sentence in sentences:
                    result += sentence['text']
        else:
            raise ValueError(
                f'call paraformer asr failed, request id: {response.get_request_id()}'
            )
        return result
