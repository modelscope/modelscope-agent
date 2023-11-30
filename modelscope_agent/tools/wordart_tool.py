import os
import time

import json
import pandas as pd
import requests
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3


class WordArtTexture(Tool):
    description = '生成艺术字纹理图片'
    name = 'wordart_texture_generation'
    parameters: list = [{
        'name': 'input.text.text_content',
        'description': 'text that the user wants to convert to WordArt',
        'required': True
    }, {
        'name': 'input.prompt',
        'description':
        'Users’ style requirements for word art may be requirements in terms of shape, color, entity, etc.',
        'required': True
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        # remote call
        self.url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/wordart/texture'
        self.token = self.cfg.get('token', os.environ.get('DASHSCOPE_API_TOKEN', ''))
        assert self.token != '', 'dashscope api token must be acquired with wordart'

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
        remote_parsed_input = json.dumps(
            self._remote_parse_input(*args, **kwargs))
        origin_result = None
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
                    url=self.url,
                    headers=headers,
                    data=remote_parsed_input)

                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))

                self.final_result = self._parse_output(
                    origin_result, remote=True)
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
                # f the key does not contain ".", directly store the key-value pair into restored_dict
                restored_dict[key] = value
            kwargs = restored_dict
            kwargs['model'] = 'wordart-texture'
        print('传给tool的参数：', kwargs)
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

    def get_wordart_result(self):
        try:
            result = self.get_result()
            print(result)
            while True:
                result_data = result.get('result', {})
                output = result_data.get('output', {})
                task_status = output.get('task_status', '')

                if task_status == 'SUCCEEDED':
                    print('任务已完成')
                    return result

                elif task_status == 'FAILED':
                    raise ('任务失败')
                else:
                    # 继续轮询，等待一段时间后再次调用
                    time.sleep(1)  # 等待 1 秒钟
                    result = self.get_result()

        except Exception as e:
            print('get Remote Error:', str(e))
