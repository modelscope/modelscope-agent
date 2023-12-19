import os
import time

import json
import requests
from modelscope_agent.tools.localfile2url_utils.localfile2url import \
    get_upload_url
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


class StyleRepaint(Tool):
    description = '调用style_repaint api处理图片'
    name = 'style_repaint'
    parameters: list = [{
        'name': 'input.image_path',
        'description': '用户上传的照片的相对路径',
        'required': True
    }, {
        'name': 'input.style_index',
        'description': '想要生成的风格化类型索引：\
            0 复古漫画 \
            1 3D童话  \
            2 二次元  \
            3 小清新  \
            4 未来科技 \
            5 3D写实 \
            用户输入数字指定风格',
        'required': True
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        # remote call
        self.url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation'
        self.token = self.cfg.get('token',
                                  os.environ.get('DASHSCOPE_API_KEY', ''))
        assert self.token != '', 'dashscope api token must be acquired'

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

    def __call__(self, *args, **kwargs):
        remote_parsed_input = self._remote_parse_input(*args, **kwargs)
        remote_parsed_input['input']['style_index'] = int(
            remote_parsed_input['input']['style_index'])
        remote_parsed_input = json.dumps(remote_parsed_input)
        origin_result = None
        retry_times = MAX_RETRY_TIMES
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

                response = requests.request(
                    'POST',
                    url=self.url,
                    headers=headers,
                    data=remote_parsed_input)

                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))

                self.final_result = self._parse_output(
                    origin_result, remote=True)
                return self.get_stylerepaint_result()
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
        image_path = kwargs['input'].pop('image_path', None)
        if image_path and image_path.endswith(('.jpeg', '.png', '.jpg')):
            # 生成 image_url，然后设置到 kwargs['input'] 中
            # 复用dashscope公共oss
            image_path = f'file://{os.path.join(WORK_DIR,image_path)}'
            print('wwwww', image_path)
            image_url = get_upload_url(
                model='qwen-vl-plus',
                upload_path=image_path,
                api_key=os.environ.get('DASHSCOPE_API_KEY', ''))
            print('wwww', image_url)
            kwargs['input']['image_url'] = image_url
        else:
            raise ValueError('请先上传一张正确格式的图片')
        kwargs['model'] = 'wanx-style-repaint-v1'
        print('传给style_repaint tool的参数：', kwargs)
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

    def get_stylerepaint_result(self):
        try:
            result = self.get_result()
            while True:
                result_data = result.get('result', {})
                output = result_data.get('output', {})
                task_status = output.get('task_status', '')

                if task_status == 'SUCCEEDED':
                    print('任务已完成')
                    # 取出result里url的部分，提高url图片展示稳定性
                    output_url = self._parse_output(
                        result['result']['output']['results'][0])
                    return output_url

                elif task_status == 'FAILED':
                    raise Exception(output.get('meaasge', '任务失败，请重试'))
                else:
                    # 继续轮询，等待一段时间后再次调用
                    time.sleep(0.5)  # 等待 0.5 秒钟
                    result = self.get_result()
                    print(f'Running:{result}')

        except Exception as e:
            print('get Remote Error:', str(e))
