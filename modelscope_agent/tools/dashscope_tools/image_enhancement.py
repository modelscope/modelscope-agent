import os
import time

import json
import requests
from modelscope_agent.constants import BASE64_FILES, ApiNames
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
        try:
            token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
            params['token'] = token
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        # 对入参格式调整和补充，比如解开嵌套的'.'连接的参数，还有导入你默认的一些参数，
        # 比如model，参考下面的_remote_parse_input函数。
        if BASE64_FILES in kwargs:
            params[BASE64_FILES] = kwargs.pop(BASE64_FILES)
        remote_parsed_input = self._parse_input(**params)
        remote_parsed_input['model'] = 'wanx-image-enhancement-v1'
        print('The parameteres pass to image enhancement:', kwargs)
        remote_parsed_input = json.dumps(remote_parsed_input)

        url = kwargs.get(
            'url',
            'https://dashscope.aliyuncs.com/api/v1/services/enhance/image-enhancement/generation'
        )
        retry_times = MAX_RETRY_TIMES

        # 参考api详情，确定headers参数
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
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
                return self.get_phantom_result(token)
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def get_phantom_result(self, token: str):
        try:
            result = self._get_task_result(token)
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
                result = self._get_task_result(token)
        except Exception as e:
            print('get request Error:', str(e))
