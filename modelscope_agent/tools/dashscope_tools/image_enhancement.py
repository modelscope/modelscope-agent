import os
import time

import json
import requests
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import register_tool
from modelscope_agent.utils.utils import get_api_key, get_upload_url
from requests.exceptions import RequestException, Timeout

from .style_repaint import StyleRepaint

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('image_enhancement')
class ImageEnhancement(StyleRepaint):
    """
        parameters是需要传入api tool的参数，通过api详情获取需要哪些必要入参
        其中每一个参数都是一个字典，包含name，description，required三个字段
        当api详情里的入参是一个嵌套object时，写成如下这种用'.'连接的格式。
    """

    description = '追影-放大镜'  # 对这个tool的功能描述
    name = 'image_enhancement'  # tool name
    parameters: list = [{
        'name': 'input.image_path',
        'type': 'string',
        'description': '输入的待增强图片的本地相对路径',
        'required': True
    }, {
        'name': 'parameters.upscale',
        'type': 'int',
        'description': '选择需要超分的倍率，可选择1、2、3、4',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        # 对入参格式调整和补充，比如解开嵌套的'.'连接的参数，还有导入你默认的一些参数，
        # 比如model，参考下面的_remote_parse_input函数。

        remote_parsed_input = json.dumps(self._remote_parse_input(**params))

        url = kwargs.get(
            'url',
            'https://dashscope.aliyuncs.com/api/v1/services/enhance/image-enhancement/generation'
        )
        retry_times = MAX_RETRY_TIMES
        try:
            self.token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')
        # 参考api详情，确定headers参数
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}',
            'X-DashScope-Async': 'enable'
        }
        # 解析oss
        headers['X-DashScope-OssResourceResolve'] = 'enable'

        while retry_times:
            retry_times -= 1
            try:
                # requests请求
                response = requests.post(
                    url=url, headers=headers, data=remote_parsed_input)

                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))
                # self._parse_output是基础类Tool对output结果的一个格式调整，你可                  # 以在这里按需调整返回格式
                self.final_result = origin_result

                # 下面是对异步api的额外get result操作，同步api可以直接得到结果的，                  # 这里返回final_result即可。
                return self.get_phantom_result()
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def _remote_parse_input(self, *args, **kwargs):
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
                # f the key does not contain ".", directly store the key-value pair into restored_dict
                restored_dict[key] = value
        kwargs = restored_dict

        image_path = kwargs['input'].pop('image_path', None)
        if image_path and image_path.endswith(
            ('.jpeg', '.png', '.jpg', '.bmp')):  # noqa E125
            # 生成 image_url，然后设置到 kwargs['input'] 中
            # 复用dashscope公共oss
            image_path = f'file://{os.path.join(WORK_DIR, image_path)}'
            image_url = get_upload_url(
                model='phantom',  # The default setting here is "style_repaint".
                file_to_upload=image_path,
                api_key=os.environ.get('DASHSCOPE_API_KEY', ''))
            kwargs['input']['image_url'] = image_url
        else:
            raise ValueError('请先上传一张正确格式的图片')

        kwargs['model'] = 'wanx-image-enhancement-v1'
        print('传给tool的参数:', kwargs)
        return kwargs

    def get_phantom_result(self):
        try:
            result = self.get_result()
            while True:
                result_data = result
                output = result_data.get('output', {})
                task_status = output.get('task_status', '')

                if task_status == 'SUCCEEDED':
                    print('任务已完成')
                    # output_url = self._parse_output(result['result']['output']['result_url'])
                    output_url = result['output']['result_url']
                    return f'![IMAGEGEN]({output_url})'

                elif task_status in ['FAILED', 'ERROR']:
                    raise ('任务失败')

                # 继续轮询，等待一段时间后再次调用
                time.sleep(1)  # 等待 1 秒钟
                result = self.get_result()
        except Exception as e:
            print('get request Error:', str(e))
