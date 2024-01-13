from typing import Union

import json
from modelscope_agent.tools import register_tool
from modelscope_agent.tools.utils.output_wrapper import VideoWrapper

from modelscope.utils.constant import Tasks
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

    task = Tasks.text_to_video_synthesis
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/text-to-video-synthesis'

    def _remote_call(self, params: str, **kwargs) -> str:
        result = super()._remote_call(params, **kwargs)
        video = result['Data']['output_video']
        return VideoWrapper(video)

    def _local_call(self, params: dict, **kwargs) -> str:
        result = super()._local_call(params, **kwargs)
        result = json.loads(result)
        video = result['output_video']
        return video

    def _verify_args(self, params: str) -> Union[str, dict]:
        params = super()._verify_args(params)
        params = {'input': {'text': params['text']}}
        return params
