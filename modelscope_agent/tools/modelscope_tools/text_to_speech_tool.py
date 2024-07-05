import json5
from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.utils.output_wrapper import AudioWrapper

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


@register_tool('speech-generation')
class TexttoSpeechTool(ModelscopePipelineTool):
    default_model = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
    description = '文本转语音服务，将文字转换为自然而逼真的语音，可配置男声/女声'
    name = 'speech-generation'
    parameters: list = [{
        'name': 'input',
        'description': '要转成语音的文本',
        'required': True,
        'type': 'string'
    }, {
        'name': 'voice',
        'description':
        '允许的声音类型：zhitian_emo（女声），zhiyan_emo（少女声），zhizhe_emo（男声），zhibei_emo（男童声）。',
        'required': True,
        'type': 'string'
    }]
    task = Tasks.text_to_speech
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/speech_sambert-hifigan_tts_zh-cn_16k'

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        if result['Code'] != 200:
            print('speech_generation error: ', result)
            return None
        audio = result['Data']['output_wav']
        return str(AudioWrapper(audio, **kwargs))

    def _verify_args(self, params: str):
        # override the args
        params_json = super()._verify_args(params)
        params_json['parameters'] = {'voice': params_json.pop('voice')}
        return params_json
