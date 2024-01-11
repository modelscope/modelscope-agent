import os
import re

import cv2
import dashscope
import json
from dashscope import ImageSynthesis
from modelscope_agent.tools.base import BaseTool, register_tool


@register_tool('image_gen')
class TextToImageTool(BaseTool):
    description = 'AI绘画（图像生成）服务，输入文本描述和图像分辨率，返回根据文本信息绘制的图片URL。'
    name = 'image_gen'
    parameters: list = [{
        'name': 'text',
        'description': '详细描述了希望生成的图像具有什么内容，例如人物、环境、动作等细节描述',
        'required': True,
        'type': 'string'
    }, {
        'name': 'resolution',
        'description':
        '格式是 数字*数字，表示希望生成的图像的分辨率大小，选项有[1024*1024, 720*1280, 1280*720]',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        if params['resolution'] in ['1024*1024', '720*1280', '1280*720']:
            resolution = params['resolution']
        else:
            resolution = '1280*720'

        prompt = params['text']
        if prompt is None:
            return None
        seed = kwargs.get('seed', None)
        model = kwargs.get('model', 'wanx-v1')
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

        response = ImageSynthesis.call(
            model=model,
            prompt=prompt,
            n=1,
            size=resolution,
            steps=10,
            seed=seed)
        image_url = response.output['results'][0]['url']
        return f'![IMAGEGEN]({image_url})'
