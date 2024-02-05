import os

import numpy as np
from modelscope_agent.tools.utils.output_wrapper import (AudioWrapper,
                                                         ImageWrapper,
                                                         OutputWrapper,
                                                         VideoWrapper)
from PIL import Image


def test_audio_wrapper():
    audio = b'an binary audio sequence'
    audio = AudioWrapper(audio)

    assert isinstance(audio.raw_data, bytes)
    assert os.path.exists(audio.path)


def test_image_wrapper():
    # generate a random image with numpy array
    img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    img = ImageWrapper(img)

    assert isinstance(img.raw_data, Image.Image)
    assert os.path.exists(img.path)


def test_video_wrapper():
    # TODO: make sure input, output follow the expectation
    pass
