import os
import time

import json
import pandas as pd
import requests
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.tools.localfile2url_utils.localfile2url import \
    get_upload_url
from pydantic import ValidationError
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('phantom_image_enhancement')
class Phantom(BaseTool):  # 继承基础类Tool，新建一个继承类
    description = '追影-放大镜'  # 对这个tool的功能描述
    name = 'phantom_image_enhancement'  # tool name
    """
    parameters是需要传入api tool的参数，通过api详情获取需要哪些必要入参
    其中每一个参数都是一个字典，包含name，description，required三个字段
    当api详情里的入参是一个嵌套object时，写成如下这种用'.'连接的格式。
    """
    parameters: list = [{
        'name': 'input.image_path',
        'description': '输入的待增强图片的本地相对路径',
        'required': True
    }, {
        'name': 'parameters.upscale',
        'description': '选择需要超分的倍率，可选择1、2、3、4',
        'required': False
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})  # cfg注册见下一条说明，这里是通过name找到对应的cfg
        # api url
        self.url = 'https://dashscope.aliyuncs.com/api/v1/services/enhance/image-enhancement/generation'
        # api token，可以选择注册在下面的cfg里，也可以选择将'API_TOKEN'导入环境变量
        self.token = self.cfg.get('token',
                                  os.environ.get('DASHSCOPE_API_KEY', ''))
        assert self.token != '', 'dashscope api token must be acquired'
        # 验证，转换参数格式，保持即可
        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)

    # 调用api操作函数，params里是llm根据上面的parameters说明得到的对应参数
    def __call__(self, params, **kwargs) -> str:
        # 对入参格式调整和补充，比如解开嵌套的'.'连接的参数，还有导入你默认的一些参数，
        # 比如model，参考下面的_remote_parse_input函数。
        params = self._verify_args(params)

        remote_parsed_input = json.dumps(self._remote_parse_input(**params))
        origin_result = None
        retry_times = MAX_RETRY_TIMES

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
                    url=self.url, headers=headers, data=remote_parsed_input)

                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))
                # self._parse_output是基础类Tool对output结果的一个格式调整，你可                  # 以在这里按需调整返回格式
                self.final_result = self._parse_output(
                    origin_result, remote=True)
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

    def _remote_parse_input(self, **kwargs):
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
        if image_path and \
           image_path.endswith(('.jpeg', '.png', '.jpg', '.bmp')):
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

    def get_result(self):
        result_data = json.loads(json.dumps(self.final_result['result']))
        if 'task_id' in result_data['output']:
            task_id = result_data['output']['task_id']
        get_url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
        get_header = {'Authorization': f'Bearer {self.token}'}

        origin_result = None
        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                response = requests.request(
                    'GET', url=get_url, headers=get_header)
                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))

                get_result = self._parse_output(origin_result, remote=True)
                return get_result
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def get_phantom_result(self):
        try:
            result = self.get_result()
            print(result)
            while True:
                result_data = result.get('result', {})
                output = result_data.get('output', {})
                task_status = output.get('task_status', '')

                # payload = result_data.get('payload', {})
                # output = payload.get('output', {})
                # res = output.get('res', {})

                if task_status == 'SUCCEEDED':
                    print('任务已完成')
                    # output_url = self._parse_output(result['result']['output']['result_url'])
                    output_url = {}
                    output_url['result'] = {}
                    output_url['result']['url'] = result['result']['output'][
                        'result_url']
                    # print(output_url)
                    print(output_url)
                    return output_url

                elif task_status in ['FAILED', 'ERROR']:
                    raise ('任务失败')

                # 继续轮询，等待一段时间后再次调用
                time.sleep(1)  # 等待 1 秒钟
                result = self.get_result()
        except Exception as e:
            print('get request Error:', str(e))
