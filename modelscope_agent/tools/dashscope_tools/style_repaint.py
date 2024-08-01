import os
import time

import json
import requests
from modelscope_agent.constants import BASE64_FILES, LOCAL_FILE_PATHS, ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import get_api_key, get_upload_url
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('style_repaint')
class StyleRepaint(BaseTool):
    description = '调用style_repaint api处理图片'
    name = 'style_repaint'
    parameters: list = [{
        'name': 'input.image_path',
        'description': '用户上传的照片的相对路径',
        'required': True,
        'type': 'string'
    }, {
        'name': 'input.style_index',
        'description': '想要生成的风格化类型索引：\
            0 复古漫画 \
            1 3D童话  \
            2 二次元  \
            3 小清新  \
            4 未来科技 \
            5 国画古风 \
            6 将军百战 \
            7 炫彩卡通 \
            8 清雅国风 \
            9 喜迎新年 \
            用户输入数字指定风格',
        'required': True,
        'type': 'int'
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
        if BASE64_FILES in kwargs:
            params[BASE64_FILES] = kwargs.pop(BASE64_FILES)

        remote_parsed_input = self._parse_input(**params)
        remote_parsed_input['model'] = 'wanx-style-repaint-v1'
        print('传给style_repaint tool的参数：', kwargs)

        try:
            remote_parsed_input['input']['style_index'] = int(
                remote_parsed_input['input']['style_index'])
        except ValueError:
            raise ValueError(
                'Please reselect the style index or the corresponding style introduction'
            )
        remote_parsed_input = json.dumps(remote_parsed_input)
        url = kwargs.get(
            'url',
            'https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation'
        )

        retry_times = MAX_RETRY_TIMES
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

                response = requests.request(
                    'POST', url=url, headers=headers, data=remote_parsed_input)

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
        kwargs = super()._parse_files_input(*args, **kwargs)

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
            # make sure the image_path is a valid image file we only get the name of the file
            image_path = image_path.split('/')[-1]
            # 生成 image_url，然后设置到 kwargs['input'] 中
            # 复用dashscope公共oss
            if LOCAL_FILE_PATHS not in kwargs or kwargs[
                    LOCAL_FILE_PATHS] == {}:
                image_path = f'file://{os.path.join(WORK_DIR,image_path)}'
            else:
                image_path = f'file://{kwargs[LOCAL_FILE_PATHS][image_path]}'

            image_url = get_upload_url(
                model=
                'style_repaint',  # The default setting here is "style_repaint".
                file_to_upload=image_path,
                api_key=kwargs['token'])
            kwargs['input']['image_url'] = image_url
        else:
            raise ValueError('请先上传一张正确格式的图片')

        return kwargs

    def _get_task_result(self, token: str):
        if 'task_id' in self.final_result['output']:
            task_id = self.final_result['output']['task_id']
        get_url = f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
        get_header = {'Authorization': f'Bearer {token}'}

        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                response = requests.request(
                    'GET', url=get_url, headers=get_header)
                if response.status_code != requests.codes.ok:
                    response.raise_for_status()
                origin_result = json.loads(response.content.decode('utf-8'))

                get_result = origin_result
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

    def _get_dashscope_image_result(self, token: str):
        try:
            result = self._get_task_result(token)
            while True:
                result_data = result
                output = result_data.get('output', {})
                task_status = output.get('task_status', '')

                if task_status == 'SUCCEEDED':
                    print('任务已完成')
                    # 取出result里url的部分，提高url图片展示稳定性
                    output_url = result['output']['results'][0]['url']
                    return f'![IMAGEGEN]({output_url})'

                elif task_status == 'FAILED':
                    raise Exception(output.get('message', '任务失败，请重试'))
                else:
                    # 继续轮询，等待一段时间后再次调用
                    time.sleep(0.5)  # 等待 0.5 秒钟
                    result = self._get_task_result(token)
                    print(f'Running:{result}')

        except Exception as e:
            raise Exception('get Remote Error:', str(e))
