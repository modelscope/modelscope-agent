import os

import cv2
import dashscope
from dashscope import ImageSynthesis
from modelscope_agent.output_wrapper import ImageWrapper

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class TextToImageTool(ModelscopePipelineTool):
    default_model = 'AI-ModelScope/stable-diffusion-xl-base-1.0'
    description = '图像生成服务，针对文本输入，生成对应的图片'
    name = 'modelscope_image-generation'
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的文本信息',
        'required': True
    }]
    model_revision = 'v1.0.0'
    task = Tasks.text_to_image_synthesis

    def _remote_parse_input(self, *args, **kwargs):
        return {'input': {'text': kwargs['text']}}

    def _local_parse_input(self, *args, **kwargs):

        text = kwargs.pop('text', '')

        parsed_args = ({'text': text}, )

        return parsed_args, {}

    def _parse_output(self, origin_result, remote=True):
        if not remote:
            image = cv2.cvtColor(origin_result['output_imgs'][0],
                                 cv2.COLOR_BGR2RGB)
        else:
            image = origin_result['output_img']

        return {'result': ImageWrapper(image)}
