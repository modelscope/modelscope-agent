import os
import json

import nls
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.tools.utils.output_wrapper import AudioWrapper
from modelscope_agent.utils.utils import get_api_key


WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('sambert_tts')
class CosyvoiceTtsTool(BaseTool):
    description = 'CosyVoice语音合成服务，将文本转成语音'
    name = 'cosyvoice_tts'
    parameters: list = [{
        'name': 'text',
        'description': '需要转成语音的文本',
        'required': True,
        'type': 'string'
    }, {
        'name': 'voice',
        'description': '音色',
        'required': True,
        'type': 'string'
    }, {
        'name': 'save_file',
        'description': '保存文件路径',
        'required': False,
        'type': 'string'
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})

        self.access_key_id = self.cfg.get(
            'aliyun_access_key_id',
            os.environ.get('ALIYUN_ACCESS_KEY_ID')
        )
        self.access_key_secret = self.cfg.get(
            'aliyun_access_key_secret',
            os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
        )
        self.app_key = self.cfg.get(
            'aliyun_app_key',
            os.environ.get('ALIYUN_APP_KEY')
        )
        if self.access_key_id is None:
            raise ValueError('Please set valid ALIYUN_ACCESS_KEY_ID!')
        if self.access_key_secret is None:
            raise ValueError('Please set valid ALIYUN_ACCESS_KEY_SECRET!')
        if self.app_key is None:
            raise ValueError('Please set valid ALIYUN_APP_KEY!')

        super().__init__(cfg)
        self.setup_token()

    def setup_token(self):
        client = AcsClient(
            self.access_key_id,
            self.access_key_secret,
           "cn-shanghai"
        )
        request = CommonRequest()
        request.set_method('POST')
        request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
        request.set_version('2019-02-28')
        request.set_action_name('CreateToken')

        try: 
            response = client.do_action_with_exception(request)
            jss = json.loads(response)
            if 'Token' in jss and 'Id' in jss['Token']:
                token = jss['Token']['Id']
                self.token = token
        except Exception as e:
            import traceback
            raise RuntimeError(
                f'Request token failed with error: {e}, with detail {traceback.format_exc()}'
            )


    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        voice = params['voice']
        aliyun_access_key_id = get_api_key(ApiNames.aliyun_access_key_id,
                                           self.access_key_id, **kwargs)
        aliyun_access_key_secret = get_api_key(ApiNames.aliyun_access_key_secret,
                                               self.access_key_secret, **kwargs)
        aliyun_app_key = get_api_key(ApiNames.aliyun_app_key,
                                     self.app_key, **kwargs)
        if aliyun_access_key_id != self.access_key_id:
            self.access_key_id = aliyun_access_key_id
            self.access_key_secret = aliyun_access_key_secret
            self.app_key = aliyun_app_key
            self.setup_token()

        tts_text = params['text']
        wav_file = params.get('save_file', None)
        return_data = b''
        if wav_file is None:
            writer = None
        else:
            writer = open(wav_file, 'wb')

        def write_data(data, *args):
            nonlocal return_data
            return_data += data
            if writer is not None:
                writer.write(data)

        def raise_error(error, *args):
            raise RuntimeError(
                f'Synthesizing speech failed with error: {error}'
            )

        def close_file(*args):
            if writer is not None:
                writer.close()
        
        sdk = nls.NlsStreamInputTtsSynthesizer(
            url="wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
            token=self.token,
            appkey=self.app_key,
            on_data=write_data,
            on_error=raise_error,
            on_close=close_file,
        )
        sdk.startStreamInputTts(
            voice=voice,
            sample_rate=16000,
            aformat="wav"
        )
        sdk.sendStreamInputTts(
            tts_text,
        )
        sdk.stopStreamInputTts()

        return return_data
