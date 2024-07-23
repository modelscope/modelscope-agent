import os

import dashscope
from dashscope import MultiModalConversation
from modelscope_agent.constants import BASE64_FILES, LOCAL_FILE_PATHS, ApiNames
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
        'name': 'image_file_paths',
        'description': '用户上传的照片的相对路径,如果是多图的话，用逗号隔开。',
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
        if BASE64_FILES in kwargs:
            params[BASE64_FILES] = kwargs.pop(BASE64_FILES)
        try:
            token = get_api_key(ApiNames.dashscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        input_params = self._parse_input(**params)
        """Sample of use local file.
        linux&mac file schema: file:///home/images/test.png
        windows file schema: file://D:/images/abc.png
        """

        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                content = []
                for local_file_path in input_params['image_file_paths']:
                    if local_file_path.lower().endswith(
                        ('.jpeg', '.png', '.jpg')):  # noqa E125
                        content.append({'image': local_file_path})
                if len(content) > 0:
                    content.append({'text': params['text']})
                    messages = [{
                        'role':
                        'system',
                        'content': [{
                            'text': 'You are a helpful assistant.'
                        }]
                    }, {
                        'role': 'user',
                        'content': content
                    }]
                    response = MultiModalConversation.call(
                        model='qwen-vl-plus', messages=messages, api_key=token)
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

    def _parse_input(self, *args, **kwargs):
        kwargs = super()._parse_files_input(*args, **kwargs)
        # convert image_file_paths from string to list
        image_file_paths = kwargs['image_file_paths']
        image_file_paths = image_file_paths.split(',')

        # current paths are deducted from llm, only contains basename, so need to add WORK_DIR
        # convert image_file_paths to a valid local file path
        if LOCAL_FILE_PATHS not in kwargs or kwargs[LOCAL_FILE_PATHS] == {}:
            # if no local file path exists, only a name of file paths exist
            for i, image_file_path in enumerate(image_file_paths):
                image_file_paths[i] = os.path.join(WORK_DIR, image_file_path)
        else:
            # if local file exists
            for i, image_file_path in enumerate(image_file_paths):
                if image_file_path not in kwargs['local_file_paths']:
                    raise ValueError(f'file {image_file_path} not exists')
                else:
                    image_file_paths[i] = kwargs[LOCAL_FILE_PATHS][
                        image_file_path]

        # convert to file:// schema for requests
        for i, image_file_path in enumerate(image_file_paths):
            image_file_paths[i] = f'file://{image_file_path}'
        kwargs['image_file_paths'] = image_file_paths
        print(
            f'传给qwen_vl tool的参数, paths:{kwargs["image_file_paths"]}, '
            f'text: {kwargs["text"]}, base64_images: {"local_file_paths" in kwargs}'
        )
        return kwargs
