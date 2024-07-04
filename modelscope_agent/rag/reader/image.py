import os
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument


class ImageToTextParser:

    def __init__(self, model: str = ''):
        self.model = model

    @abstractmethod
    def generate(self, image: Path, prompt: str) -> str:
        pass


class OpenaiAPIParser(ImageToTextParser):

    def __init__(
        self,
        model: str = 'qwen-vl-max',
        base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key: str = os.getenv('DASHSCOPE_API_KEY', '')):  # noqa
        self.base_url = base_url
        self.api_key = api_key
        super().__init__(model)

    def generate(self, image: Path, prompt: str, **kwargs) -> str:
        import base64
        import mimetypes
        from openai import OpenAI

        if self.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1' and not len(
                self.api_key):
            self.api_key = kwargs.get('api_key',
                                      os.getenv('DASHSCOPE_API_KEY', ''))
        assert len(self.api_key), 'api_key is not set.'
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        image_path = image.__str__()
        mime_type, _ = mimetypes.guess_type(image_path)

        # 校验MIME类型为支持的图片格式, 限制图片大小在5M内
        if mime_type and mime_type.startswith(
                'image') and os.path.getsize(image_path) < 5000000:
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read())
                encoded_image_str = encoded_image.decode('utf-8')
                data_uri_prefix = f'data:{mime_type};base64,'
                encoded_image_str = data_uri_prefix + encoded_image_str

                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        'role':
                        'user',
                        'content': [{
                            'type': 'text',
                            'text': prompt
                        }, {
                            'type': 'image_url',
                            'image_url': {
                                'url': encoded_image_str
                            }
                        }]
                    }],
                    top_p=0.8,
                    stream=True,
                    stream_options={'include_usage': True})
            res = ''
            for chunk in completion:
                if len(chunk.choices) > 0 and hasattr(
                        chunk.choices[0].delta,
                        'content') and chunk.choices[0].delta.content:
                    res += chunk.choices[0].delta.content
            return res
        else:
            print(
                'image file type unsupported, image file size exceeds 5M or file not found.'
            )
            return ''


class OcrParser(ImageToTextParser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import tensorflow as tf
            import pyclipper
            import shapely
            import tf_slim
        except Exception as e:
            print(
                f'Using OcrParser requires the installation of tensorflow, tf_slim, pyclipper and shapely '
                f'dependencies, which you can install using `pip install tensorflow tf_slim pyclipper '
                f'shapely`. Error details: {e}')
            raise ImportError(e)
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.ocr_detection = pipeline(
            Tasks.ocr_detection,
            model='iic/cv_resnet18_ocr-detection-db-line-level_damo')
        self.ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-general_damo')

    def generate(self, image: Path, prompt: str = '', **kwargs) -> str:
        import cv2
        import math
        import numpy as np

        # scripts for crop images
        def crop_image(img, position):

            def distance(x1, y1, x2, y2):
                return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

            position = position.tolist()
            for i in range(4):
                for j in range(i + 1, 4):
                    if (position[i][0] > position[j][0]):
                        tmp = position[j]
                        position[j] = position[i]
                        position[i] = tmp
            if position[0][1] > position[1][1]:
                tmp = position[0]
                position[0] = position[1]
                position[1] = tmp

            if position[2][1] > position[3][1]:
                tmp = position[2]
                position[2] = position[3]
                position[3] = tmp

            x1, y1 = position[0][0], position[0][1]
            x2, y2 = position[2][0], position[2][1]
            x3, y3 = position[3][0], position[3][1]
            x4, y4 = position[1][0], position[1][1]

            corners = np.zeros((4, 2), np.float32)
            corners[0] = [x1, y1]
            corners[1] = [x2, y2]
            corners[2] = [x4, y4]
            corners[3] = [x3, y3]

            img_width = distance((x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2,
                                 (y2 + y3) / 2)
            img_height = distance((x1 + x2) / 2, (y1 + y2) / 2, (x4 + x3) / 2,
                                  (y4 + y3) / 2)

            corners_trans = np.zeros((4, 2), np.float32)
            corners_trans[0] = [0, 0]
            corners_trans[1] = [img_width - 1, 0]
            corners_trans[2] = [0, img_height - 1]
            corners_trans[3] = [img_width - 1, img_height - 1]

            transform = cv2.getPerspectiveTransform(corners, corners_trans)
            dst = cv2.warpPerspective(img, transform,
                                      (int(img_width), int(img_height)))
            return dst

        def order_point(coor):
            arr = np.array(coor).reshape([4, 2])
            sum_ = np.sum(arr, 0)
            centroid = sum_ / arr.shape[0]
            theta = np.arctan2(arr[:, 1] - centroid[1],
                               arr[:, 0] - centroid[0])
            sort_points = arr[np.argsort(theta)]
            sort_points = sort_points.reshape([4, -1])
            if sort_points[0][0] > centroid[0]:
                sort_points = np.concatenate(
                    [sort_points[3:], sort_points[:3]])
            sort_points = sort_points.reshape([4, 2]).astype('float32')
            return sort_points

        img_path = image.__str__()
        image_full = cv2.imread(img_path)
        det_result = self.ocr_detection(image_full)
        det_result = det_result['polygons']
        res = ''
        for i in range(det_result.shape[0] - 1, -1, -1):
            pts = order_point(det_result[i])
            image_crop = crop_image(image_full, pts)
            result = self.ocr_recognition(image_crop)
            box = ','.join([str(e) for e in list(pts.reshape(-1))])
            text = result['text'][0]
            res += str({'box': box, 'text': text})
        return res


def get_image_parser(image_parser: Union[Type[ImageToTextParser],
                                         ImageToTextParser, None] = None):
    if image_parser:
        if isinstance(image_parser, ImageToTextParser):
            return image_parser
        elif isinstance(image_parser, type(ImageToTextParser)):
            try:
                return image_parser()
            except Exception as e:
                print(
                    f'Initialization of image_parser {image_parser} A failed, details: {e}'
                )
        else:
            print(
                f'image_parser {image_parser} has not supported yet. Using default image parser: dashscope qwen-vl-max.'
            )
    return OpenaiAPIParser()


class CustomImageReader(BaseReader):

    def __init__(
        self,
        image_parser: Union[Type[ImageToTextParser], ImageToTextParser,
                            None] = None,
        keep_image: bool = False,
        parse_text: bool = True,
        prompt: str = '图片的内容是什么？',
    ):
        """Init params."""
        self._parser = None
        if parse_text:
            self._parser = get_image_parser(image_parser)

        self._parse_text = parse_text
        self._keep_image = keep_image
        self._prompt = prompt

    def load_data(self,
                  file: Path,
                  extra_info: Optional[Dict] = None) -> List[Document]:
        """Parse file."""
        from PIL import Image

        from llama_index.legacy.img_utils import img_2_b64

        # load document image
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Encode image into base64 string and keep in document
        image_str: Optional[str] = None
        if self._keep_image:
            image_str = img_2_b64(image)

        # Parse image into text
        text_str: str = ''
        if self._parse_text:
            text_str = self._parser.generate(image=file, prompt=self._prompt)
        return [
            ImageDocument(
                text=text_str,
                image=image_str,
                image_path=str(file),
                metadata=extra_info or {},
            )
        ]


if __name__ == '__main__':
    fp = 'tests/samples/rag.png'
    a = OcrParser()
    res = a.generate(Path(fp), '图片的内容是什么？')
    print(res)
