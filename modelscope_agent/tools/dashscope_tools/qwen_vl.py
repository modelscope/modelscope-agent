import os

import dashscope
from dashscope import MultiModalConversation
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import get_api_key
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('qwen_vl')
class QWenVL(BaseTool):
    description = '中英文图文对话，支持图片内容识别，支持图片里中英双语的长文本识别'
    name = 'qwen_vl'
    parameters: list = [{
        'name': 'image_file_path',
        'description': '用户上传的照片的相对路径',
        'required': True,
        'type': 'string'
    }, {
        'name': 'text',
        'description': '用户针对上传图片的提问文本',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        # 检查环境变量中是否设置DASHSCOPE_API_KEY

        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        remote_parsed_input = self._remote_parse_input(**params)
        dashscope.api_key = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        """Sample of use local file.
        linux&mac file schema: file:///home/images/test.png
        windows file schema: file://D:/images/abc.png
        """
        local_file_path = f"file://{remote_parsed_input['image_file_path']}"

        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                if local_file_path.lower().endswith(('.jpeg', '.png', '.jpg')):
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
                                'text': params['text']
                            },
                        ]
                    }]
                    response = MultiModalConversation.call(
                        model='qwen-vl-plus', messages=messages)
                    return response['output']['choices'][0]['message'][
                        'content'][0]
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
            except Exception as e:
                raise ValueError(f'Error: {e}')

    def _remote_parse_input(self, *args, **kwargs):
        kwargs['image_file_path'] = os.path.join(WORK_DIR,
                                                 kwargs['image_file_path'])
        print('传给qwen_vl tool的参数：', kwargs)
        return kwargs
