import os

import dashscope
from dashscope import ImageSynthesis
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import get_api_key

MAX_RETRY_TIMES = 3


@register_tool('image_gen_lite')
class TextToImageLiteTool(BaseTool):
    description = 'AI绘画（图像生成）服务，输入文本描述和图像分辨率，返回根据文本信息绘制的图片URL，同时允许用户通过添加lora层来选择风格化的图片'
    name = 'image_gen_lite'
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
    }, {
        'name': 'lora_index',
        'description':
        '如果用户指定使用lora则使用该参数，通过选择的lora层来决定生成的图像的风格，如果用户没有制定，则默认为wanxlite1.4.5_lora_huibenlite1_20240519',
        'required': False,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        try:
            token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        if params['resolution'] in ['1024*1024', '720*1280', '1280*720']:
            resolution = params['resolution']
        else:
            resolution = '1280*720'

        prompt = params['text']
        if prompt is None:
            return None
        seed = kwargs.get('seed', None)
        lora_index = params.get('lora_index', None)
        if lora_index:
            extra_input = {'lora_index': lora_index}
            model = kwargs.get('model', 'wanx-lora-lite')
        else:
            extra_input = None
            model = kwargs.get('model', 'wanx-lite-v1')

        response = ImageSynthesis.call(
            model=model,
            prompt=prompt,
            n=1,
            size=resolution,
            steps=10,
            seed=seed,
            extra_input=extra_input,
            api_key=token)
        image_url = response.output['results'][0]['url']
        return f'![IMAGEGEN]({image_url})'
