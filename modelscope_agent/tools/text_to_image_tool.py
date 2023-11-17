import os
import re

import cv2
import dashscope
import json
from dashscope import ImageSynthesis
from modelscope_agent.output_wrapper import ImageWrapper

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


def is_multiply_format(s):
    pattern = r'^\d+\*\d+$'
    if re.match(pattern, s):
        return True
    else:
        return False


class TextToImageTool(ModelscopePipelineTool):
    default_model = 'AI-ModelScope/stable-diffusion-xl-base-1.0'
    description = 'AI绘画（图像生成）服务，输入文本描述，返回根据文本信息绘制的图片URL。'
    name = 'image_gen'
    parameters: list = [{
        'name': 'text',
        'description': '中文的文本提示词，描述了希望生成的图像具有什么内容',
        'required': True,
        'schema': {
            'type': 'string'
        }
    }, {
        'name': 'resolution',
        'description': '格式是两个数字相乘，表示希望生成的图像的分辨率',
        'required': True,
        'schema': {
            'type': 'string'
        }
    }]
    model_revision = 'v1.0.0'
    task = Tasks.text_to_image_synthesis

    # def _remote_parse_input(self, *args, **kwargs):
    #     params = {
    #         'input': {
    #             'text': kwargs['text'],
    #             'resolution': kwargs['resolution']
    #         }
    #     }
    #     if kwargs.get('seed', None):
    #         params['input']['seed'] = kwargs['seed']
    #     return params

    def _remote_call(self, *args, **kwargs):

        if 'resolution' in kwargs and is_multiply_format(kwargs['resolution']):
            resolution = kwargs['resolution']
            resolution = resolution.replace('x', '*')
        else:
            resolution = '1280*720'

        prompt = kwargs['text']
        seed = kwargs.get('seed', None)
        if prompt is None:
            return None
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
        response = ImageSynthesis.call(
            model=ImageSynthesis.Models.wanx_v1,
            prompt=prompt,
            n=1,
            size=resolution,
            steps=10,
            seed=seed)
        final_result = self._parse_output(response, remote=True)
        return final_result

    def _local_parse_input(self, *args, **kwargs):

        text = kwargs.pop('text', '')

        parsed_args = ({'text': text}, )

        return parsed_args, {}

    def _parse_output(self, origin_result, remote=True):
        if not remote:
            image = cv2.cvtColor(origin_result['output_imgs'][0],
                                 cv2.COLOR_BGR2RGB)
        else:
            image = origin_result.output['results'][0]['url']

        return {'result': ImageWrapper(image)}

    def _handle_input_fallback(self, **kwargs):
        """
        an alternative method is to parse image is that get item between { and }
        for last try

        :param fallback_text:
        :return: language, cocde
        """

        text = kwargs.get('text', None)
        fallback = kwargs.get('fallback', None)

        if text:
            return text
        elif fallback:
            try:
                text = fallback
                json_block = re.search(r'\{([\s\S]+)\}', text)  # noqa W^05
                if json_block:
                    result = json_block.group(1)
                    result_json = json.loads('{' + result + '}')
                    return result_json['text']
            except ValueError:
                return text
        else:
            return text
