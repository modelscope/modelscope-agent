import time
import json
import requests
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import get_api_key
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3

@register_tool('image_gen_lora')
class TextToImageLoraTool(BaseTool):
    description = 'AI绘画（图像生成）服务，输入文本描述和图像分辨率，返回根据文本信息绘制的图片URL，同时允许用户通过添加lora层来选择风格化的图片'
    name = 'image_gen_lora'
    parameters: list = [
        {
            'name': 'input.prompt',
            'description': '详细描述了希望生成的图像具有什么内容，例如人物、环境、动作等细节描述',
            'required': True,
            'type': 'string'
        },
        {
            'name': 'input.lora_index',
            'description': '通过选择的lora层来决定生成的图像的风格，如果用户没有制定，则默认为wanxlite1.4.5_lora_huibenlite1_20240519',
            'required': True,
            'type': 'string'
        },
        {
            'name': 'parameters.size',
            'description': '生成图像的分辨率，目前仅支持1024*1024，720*1280，1280*720三种分辨率，默认为1024*1024像素',
            'required': True,
            'type': 'string'
        },
        {
            'name': 'parameters.n',
            'description': '图片生成的数量，目前支持1～6张，如果用户没有制定，则默认为1个',
            'required': True,
            'type': 'int'
        },
        {
            'name': 'parameters.seed',
            'description': '图片生成时候的种子值，如果不提供，这里则忽略',
            'required': False,
            'type': 'int'
        }
    ]


    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        remote_parsed_input = json.dumps(self._remote_parse_input(**params))
        try:
            self.token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')
        
        retry_times = MAX_RETRY_TIMES
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}',
            'X-DashScope-Async': 'enable'
        }

        while retry_times:
            retry_times -= 1
            try:

                response = requests.request(
                    'POST',
                    url=
                    'https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis',
                    headers=headers,
                    data=remote_parsed_input)

                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))

                self.final_result = origin_result
                return self.get_wordart_result()
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
                # if the key does not contain ".", directly store the key-value pair into restored_dict
                restored_dict[key] = value
            kwargs = restored_dict
            kwargs['model'] = 'wanx-lora-lite'
        print('传给tool的参数：', kwargs)
        return kwargs
    

    def get_result(self):
        result_data = json.loads(json.dumps(self.final_result))
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
                return origin_result
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )
    

    def get_wordart_result(self):
        try:
            result = self.get_result()
            while True:
                result_data = result
                output = result_data.get('output', {})
                task_status = output.get('task_status', '')

                if task_status == 'SUCCEEDED':
                    print('任务已完成')
                    # 取出result里url的部分，提高url图片展示稳定性
                    output_url = result['output']['results'][0]['url']
                    return output_url

                elif task_status == 'FAILED':
                    raise Exception(output.get('message', '任务失败，请重试'))
                else:
                    # 继续轮询，等待一段时间后再次调用
                    time.sleep(3)  # 等待 1 秒钟
                    result = self.get_result()
                    print(f'Running:{result}')
        except Exception as e:
            print('get Remote Error:', str(e))