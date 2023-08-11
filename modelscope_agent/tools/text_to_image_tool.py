import os

import cv2
import dashscope
from dashscope import ImageSynthesis
from modelscope_agent.output_wrapper import ImageWrapper

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool

dashscope.api_key = os.getenv('dashcope_api_key')


class TextToImageTool(ModelscopePipelineTool):
    default_model = 'AI-ModelScope/stable-diffusion-v2-1'
    description = '图像生成服务，针对文本输入，生成对应的图片'
    name = 'modelscope_image-generation'
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的文本信息',
        'required': True
    }]
    model_revision = 'v1.0.0'
    task = Tasks.text_to_image_synthesis

    def _local_parse_input(self, *args, **kwargs):

        text = kwargs.pop('text', '')

        parsed_args = ({'text': text}, )

        return parsed_args, {}

    def _remote_call(self, *args, **kwargs):
        response = ImageSynthesis.call(
            model=ImageSynthesis.Models.wanx_v1,
            prompt=kwargs['text'],
            n=1,
            size='1024*1024',
            steps=10)
        final_result = self._parse_output(response, remote=True)
        return final_result

    def _parse_output(self, origin_result, remote=True):
        if not remote:
            image = cv2.cvtColor(origin_result['output_imgs'][0],
                                 cv2.COLOR_BGR2RGB)
        else:
            image = origin_result['output']['results'][0]['url']

        return {'result': ImageWrapper(image)}
