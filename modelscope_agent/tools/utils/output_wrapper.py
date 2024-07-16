import os
import tempfile
import uuid
from typing import Dict

import numpy as np
import requests
from modelscope_agent.tools.utils.oss import OssStorage
from PIL import Image
from requests.exceptions import RequestException


class OutputWrapper:
    """
    Wrapper for output of tool execution when output is image, video, audio, etc.
    In this wrapper, __repr__() is implemented to return the str representation of the output for llm.
    Each wrapper have below attributes:
        path: the path where the output is stored
        raw_data: the raw data, e.g. image, video, audio, etc. In remote mode, it should be None
    """

    def __init__(self) -> None:
        self._repr = None
        self._path = None
        self._raw_data = None
        self.oss = None

        self.root_path = os.environ.get('OUTPUT_FILE_DIRECTORY', None)
        if self.root_path and not os.path.exists(self.root_path):
            try:
                os.makedirs(self.root_path)
            except Exception:
                self.root_path = None

    def get_remote_file(self, remote_path, suffix):
        try:
            response = requests.get(remote_path)
            obj = response.content
            directory = tempfile.mkdtemp(dir=self.root_path)
            path = os.path.join(directory, str(uuid.uuid4()) + f'.{suffix}')
            with open(path, 'wb') as f:
                f.write(obj)
            return path
        except RequestException:
            return remote_path

    def _upload_and_get_oss_url(self):
        try:
            file_name = os.path.basename(self._path)
            oss_path = os.path.join('tmp', 'video', file_name)
            self.oss.upload(self._path, oss_path)
            return self.oss.get_signed_url(oss_path)
        except Exception as e:
            print(
                f'Failed to save the file to oss with error: {e}, please check the oss information'
            )
            return ''

    def __repr__(self) -> str:
        return self._repr

    @property
    def path(self):
        return self._path

    @property
    def raw_data(self):
        return self._raw_data


class ImageWrapper(OutputWrapper):
    """
    Image wrapper, raw_data is a PIL.Image
    """

    def __init__(self, image) -> None:

        super().__init__()

        if isinstance(image, str):
            if os.path.isfile(image):
                self._path = image
            else:
                self._path = self.get_remote_file(image, 'png')
            try:
                image = Image.open(self._path)
                self._raw_data = image
            except FileNotFoundError:
                # Image store in remote server when use remote mode
                raise FileNotFoundError(f'Invalid path: {image}')
        else:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image.astype(np.uint8))
                self._raw_data = image
            else:
                self._raw_data = image
            directory = tempfile.mkdtemp(dir=self.root_path)
            self._path = os.path.join(directory, str(uuid.uuid4()) + '.png')
            self._raw_data.save(self._path)

        self._repr = f'![IMAGEGEN]({self._path})'


class AudioWrapper(OutputWrapper):
    """
    Audio wrapper, raw_data is a binary file
    """

    def __init__(self, audio, **kwargs) -> None:

        super().__init__()
        if isinstance(audio, str):
            # use_tool_api should use no file, just bypass url or base64
            if 'use_tool_api' in kwargs and kwargs['use_tool_api']:
                self._path = audio
            else:
                if os.path.isfile(audio):
                    self._path = audio
                else:
                    self._path = self.get_remote_file(audio, 'wav')
                try:
                    with open(self._path, 'rb') as f:
                        self._raw_data = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f'Invalid path: {audio}')
        else:
            self._raw_data = audio
            directory = tempfile.mkdtemp(dir=self.root_path)
            self._path = os.path.join(directory, str(uuid.uuid4()) + '.wav')

            with open(self._path, 'wb') as f:
                f.write(self._raw_data)

        self._repr = f'<audio src="{self._path}"/>'


class VideoWrapper(OutputWrapper):
    """
    Video wrapper
    """

    def __init__(self, video, **kwargs) -> None:
        super().__init__()
        if isinstance(video, str):
            # use_tool_api should use no file, just bypass url or base64
            use_remote_url = False
            if 'use_tool_api' in kwargs and kwargs['use_tool_api']:
                use_remote_url = True
                if self.oss is None:
                    self.oss = OssStorage()
            else:
                try:
                    from moviepy.editor import VideoFileClip
                except Exception:
                    raise ImportError(
                        'moviepy is required when output is video, please install it first by `pip install moviepy`'
                    )

            if os.path.isfile(video):
                self._path = video
                if use_remote_url:
                    self._path = self._upload_and_get_oss_url()
            else:
                if use_remote_url:
                    self._path = video
                else:
                    self._path = self.get_remote_file(video, 'gif')

            if not use_remote_url:
                try:
                    video = VideoFileClip(self._path)

                except (ValueError, OSError):
                    raise FileNotFoundError(f'Invalid path: {video}')
        else:
            raise TypeError(
                'Current only support load from filepath when it is video')

        self._raw_data = video

        if self._path.endswith('.gif'):
            self._repr = f'![IMAGEGEN]({self._path})'
        else:
            self._repr = f'<video src="{self._path}"/>'


def get_raw_output(exec_result: Dict):
    # get rwa data of exec_result
    res = {}
    for k, v in exec_result.items():
        if isinstance(v, OutputWrapper):
            # In remote mode, raw data maybe None
            res[k] = v.raw_data or str(v)
        else:
            res[k] = v
    return res
