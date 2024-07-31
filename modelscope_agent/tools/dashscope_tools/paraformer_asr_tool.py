import os
import subprocess
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import dashscope
from modelscope_agent.constants import LOCAL_FILE_PATHS, ApiNames
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
        super().__init__(cfg)

    def call(self, params: str, **kwargs):
        from dashscope.audio.asr import Recognition
        params = self._verify_args(params)
        kwargs = super()._parse_files_input(**kwargs)

        try:
            token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        # make sure the audio_path is file name not file path
        params['audio_path'] = params['audio_path'].split('/')[-1]
        if LOCAL_FILE_PATHS not in kwargs or kwargs[LOCAL_FILE_PATHS] == {}:
            raw_audio_file = WORK_DIR + '/' + params['audio_path']
        else:
            raw_audio_file = kwargs[LOCAL_FILE_PATHS][params['audio_path']]
        if not os.path.exists(raw_audio_file):
            raise ValueError(f'audio file {raw_audio_file} not exists')
        try:
            pcm_file = os.path.join(
                WORK_DIR,
                os.path.basename(params['audio_path']).split('.')[0] + '.pcm')
            _preprocess(raw_audio_file, pcm_file)
            if not os.path.exists(pcm_file):
                raise ValueError(
                    f'convert audio to pcm file {pcm_file} failed')
            recognition = Recognition(
                model='paraformer-realtime-v1',
                format='pcm',
                sample_rate=16000,
                callback=None)
            response = recognition.call(
                pcm_file,
                api_key=token,
            )
        except Exception as e:
            import traceback
            print(
                f'call paraformer asr failed, error: {e}, and traceback {traceback.format_exc()}'
            )
            raise ValueError(f'call paraformer asr failed, error: {e}')
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
