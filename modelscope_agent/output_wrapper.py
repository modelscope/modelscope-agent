import os
import re
import tempfile
import uuid
from typing import Dict

import json
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image


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

        self.root_path = os.environ.get('OUTPUT_FILE_DIRECTORY', None)
        if not os.path.exists(self.root_path):
            try:
                os.makedirs(self.root_path)
            except Exception:
                self.root_path = None

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
            try:
                self._path = image
                image = Image.open(self._path)
                self._raw_data = image
            except FileNotFoundError:
                # Image store in remote server when use remote mode
                pass
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

    def __init__(self, audio) -> None:

        super().__init__()
        if isinstance(audio, str):
            try:
                self._path = audio
                with open(self._path, 'rb') as f:
                    self._raw_data = f.read()
            except FileNotFoundError:
                pass
        else:
            self._raw_data = audio
            directory = tempfile.mkdtemp(dir=self.root_path)
            self._path = os.path.join(directory, str(uuid.uuid4()) + '.wav')

            with open(self._path, 'wb') as f:
                f.write(self._raw_data)

        self._repr = f'<audio id=audio controls= preload=none> <source id=wav src={self._path}> </audio>'


class VideoWrapper(OutputWrapper):
    """
    Video wrapper
    """

    def __init__(self, video) -> None:

        super().__init__()
        if isinstance(video, str):
            try:
                self._path = video
                video = VideoFileClip(self._path)
                # currently, we should save video as gif, not mp4
                if not self._path.endswith('gif'):
                    directory = tempfile.mkdtemp(dir=self.root_path)
                    self._path = os.path.join(directory,
                                              str(uuid.uuid4()) + '.gif')
                    video.write_gif(self._path)
            except (ValueError, OSError):
                pass
        else:
            raise TypeError(
                'Current only support load from filepath when it is video')

        self._raw_data = video
        self._repr = f'![IMAGEGEN]({self._path})'


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


def display(llm_result: str, idx: int):
    """Display the result of each round in jupyter notebook.
    The multi-modal data will be extracted.

    Args:
        llm_result (str): llm result
        idx (int): current round
    """
    from IPython.display import display, Pretty, Image, Audio, JSON
    idx_info = '*' * 50 + f'round {idx}' + '*' * 50
    display(Pretty(idx_info))
    match_image = re.search(r'!\[IMAGEGEN\]\((.*?)\)', llm_result)
    if match_image:
        result = match_image.group(1)
        try:
            display(Image(result))
            llm_result = llm_result.replace(match_image.group(0), '')
        except Exception:
            pass

    match_audio = re.search(
        r'<audio id=audio controls= preload=none> <source id=wav src=(.*?)> <\/audio>',
        llm_result)
    if match_audio:
        result = match_audio.group(1)
        try:
            display(Audio(result))
            llm_result = llm_result.replace(match_audio.group(0), '')
        except Exception:
            pass

    match_action = re.search(
        r'<\|startofthink\|>```JSON([\s\S]*)```<\|endofthink\|>', llm_result)
    if match_action:
        result = match_action.group(1)
        try:
            json_content = json.loads(result, strict=False)
            display(JSON(json_content))
            llm_result = llm_result.replace(match_action.group(0), '')
        except Exception:
            pass

    display(Pretty(llm_result))
