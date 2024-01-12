import os
import tempfile
import uuid

from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.utils.output_wrapper import VideoWrapper

from .pipeline_tool import ModelscopePipelineTool

@register_tool('video-generation')
class TextToVideoTool(ModelscopePipelineTool):
    default_model = 'damo/text-to-video-synthesis'
    description = '视频生成服务，针对英文文本输入，生成一段描述视频'

    name = 'video-generation'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的文本信息，仅支持英文文本描述',
        'required': True,
        'type': 'string'
    }]
    API_URL = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/text-to-video-synthesis'

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        if result['Code'] != 200:
            print('video_generation error: ', result)
            return None
        print(f'result: {result}')
        video = result['Data']['output_video']
        return VideoWrapper(video)

