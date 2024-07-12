import json
import requests
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.tools.dashscope_tools.style_repaint import StyleRepaint
from modelscope_agent.utils.utils import get_api_key
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3


@register_tool('wordart_texture_generation')
class WordArtTexture(StyleRepaint):
    description = '生成艺术字纹理图片'
    name = 'wordart_texture_generation'
    parameters: list = [{
        'name': 'input.text.text_content',
        'description': 'text that the user wants to convert to WordArt',
        'required': True,
        'type': 'string'
    }, {
        'name': 'input.prompt',
        'description':
        'Users’ style requirements for word art may be requirements in terms of shape, color, entity, etc.',
        'required': True,
        'type': 'string'
    }, {
        'name': 'input.texture_style',
        'description':
        'Type of texture style;Default is "material";If not provided by the user, \
            defaults to "material".Another value is scene.',
        'required': True,
        'type': 'string'
    }, {
        'name': 'input.text.output_image_ratio',
        'description':
        'The aspect ratio of the text input image; the default is "1:1", \
            the available ratios are: "1:1", "16:9", "9:16";',
        'required': True,
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
        remote_parsed_input = json.dumps(self._parse_input(**params))

        retry_times = MAX_RETRY_TIMES
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
            'X-DashScope-Async': 'enable'
        }
        while retry_times:
            retry_times -= 1
            try:

                response = requests.request(
                    'POST',
                    url=
                    'https://dashscope.aliyuncs.com/api/v1/services/aigc/wordart/texture',
                    headers=headers,
                    data=remote_parsed_input)

                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))

                self.final_result = origin_result
                return self._get_dashscope_image_result(token)
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def _parse_input(self, *args, **kwargs):
        restored_dict = {}
        for key, value in kwargs.items():
            if '.' in key:
                # Split keys by "." and create nested dictionary structures
                keys = key.split('.')
                temp_dict = restored_dict
                for k in keys[:-1]:
                    temp_dict = temp_dict.setdefault(k, {})
                temp_dict[keys[-1]] = value
            else:
                # if the key does not contain ".", directly store the key-value pair into restored_dict
                restored_dict[key] = value
            kwargs = restored_dict
            kwargs['model'] = 'wordart-texture'
        print('传给tool的参数：', kwargs)
        return kwargs
