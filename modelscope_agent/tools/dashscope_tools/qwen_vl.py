import os
import time

import json
import requests
from dashscope import MultiModalConversation
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


class QWenVL(Tool):
    description = '调用qwen_vl api处理图片'
    name = 'qwen_vl'
    parameters: list = [{
        'name': 'image_file_path',
        'description': '用户上传的照片的相对路径',
        'required': True
    }, {
        'name': 'text',
        'description': '用户针对上传图片的提问文本',
        'required': True
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        # remote call
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
        """Sample of use local file.
        linux&mac file schema: file:///home/images/test.png
        windows file schema: file://D:/images/abc.png
        """
        local_file_path = f"file://{remote_parsed_input['image_file_path']}"

        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                if local_file_path.endswith(('.jpeg', '.png', '.jpg')):
                    messages = [{
                        'role':
                        'system',
                        'content': [{
                            'text': 'You are a helpful assistant.'
                        }]
                    }, {
                        'role':
                        'user',
                        'content': [
                            {
                                'image': local_file_path
                            },
                            {
                                'text': kwargs['text']
                            },
                        ]
                    }]
                    response = MultiModalConversation.call(
                        model='qwen-vl-plus', messages=messages)
                    final_result = self._parse_output(response)
                    return final_result
                else:
                    raise ValueError(
                        f'the file you upload: {local_file_path} is not an image file, \
                                     please upload true file and try again.')
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

    def _remote_parse_input(self, *args, **kwargs):
        kwargs['image_file_path'] = os.path.join(WORK_DIR,
                                                 kwargs['image_file_path'])
        print('传给qwen_vl tool的参数：', kwargs)
        return kwargs
