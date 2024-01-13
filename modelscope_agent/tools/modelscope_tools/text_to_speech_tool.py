import json
from modelscope_agent.tools import register_tool
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
        'description': '用户身份',
        'required': True,
        'type': 'string'
    }]

    task = Tasks.text_to_speech
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/speech_sambert-hifigan_tts_zh-cn_16k'

    def _remote_call(self, params: str, **kwargs) -> str:
        result = super()._remote_call(params, **kwargs)

        audio = result['Data']['output_wav']
        return AudioWrapper(audio)

    def _local_call(self, params: dict, **kwargs) -> str:
        result = super()._local_call(params, **kwargs)
        result = json.loads(result)
        audio = result['output_wav']
        return audio
