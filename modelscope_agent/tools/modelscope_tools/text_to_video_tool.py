from typing import Union
from modelscope_agent.tools.utils.output_wrapper import VideoWrapper
from .pipeline_tool import ModelscopePipelineTool

class TextToVideoTool(ModelscopePipelineTool):
    default_model = 'damo/text-to-video-synthesis'
    description = '视频生成服务，针对英文文本输入，生成一段描述视频；如果是中文输入同时依赖插件modelscope_text-translation-zh2en翻译成英文'
    name = 'video-generation'
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的文本信息',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        video = result['Data']['output_video']
        return VideoWrapper(video)
    
    def _verify_args(self, params: str) -> Union[str, dict]:
        params = super()._verify_args(params)
        params = {'input': {'text': params['text']}}
        return params

