import os
import tempfile
import uuid

from modelscope_agent.output_wrapper import VideoWrapper

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class TextToVideoTool(ModelscopePipelineTool):
    default_model = 'damo/text-to-video-synthesis'
    description = '视频生成服务，针对英文文本输入，生成一段描述视频；如果是中文输入同时依赖插件modelscope_text-translation-zh2en翻译成英文'

    name = 'video-generation'
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的文本信息',
        'required': True
    }]
    task = Tasks.text_to_video_synthesis

    def _remote_parse_input(self, *args, **kwargs):
        return {'input': {'text': kwargs['text']}}

    def _local_parse_input(self, *args, **kwargs):

        text = kwargs.pop('text', '')
        directory = tempfile.mkdtemp()
        file_path = os.path.join(directory, str(uuid.uuid4()) + '.mp4')

        parsed_args = ({'text': text}, )
        parsed_kwargs = {'output_video': file_path}

        return parsed_args, parsed_kwargs

    def _parse_output(self, origin_result, remote=True):

        video = origin_result['output_video']
        return {'result': VideoWrapper(video)}
