from modelscope_agent.output_wrapper import AudioWrapper

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class TexttoSpeechTool(ModelscopePipelineTool):
    default_model = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
    description = '文本转语音服务，将文字转换为自然而逼真的语音，可配置男声/女声'
    name = 'speech-generation'
    parameters: list = [{
        'name': 'input',
        'description': '要转成语音的文本',
        'required': True
    }, {
        'name': 'gender',
        'description': '用户身份',
        'required': True
    }]
    task = Tasks.text_to_speech

    def _local_parse_input(self, *args, **kwargs):
        if 'gender' not in kwargs:
            kwargs['gender'] = 'man'
        voice = 'zhizhe_emo' if kwargs['gender'] == 'man' else 'zhiyan_emo'
        kwargs['voice'] = voice
        if 'text' in kwargs and 'input' not in kwargs:
            kwargs['input'] = kwargs['text']
            kwargs.pop('text')
        kwargs.pop('gender')
        return args, kwargs

    def _remote_parse_input(self, *args, **kwargs):
        if 'gender' not in kwargs:
            kwargs['gender'] = 'man'
        voice = 'zhizhe_emo' if kwargs['gender'] == 'man' or kwargs[
            'gender'] == 'male' else 'zhiyan_emo'
        kwargs['parameters'] = {'voice': voice}
        kwargs.pop('gender')
        return kwargs

    def _parse_output(self, origin_result, remote=True):

        audio = origin_result['output_wav']
        return {'result': AudioWrapper(audio)}
