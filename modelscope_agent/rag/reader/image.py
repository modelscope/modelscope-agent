import os
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
from PIL.PngImagePlugin import PngImageFile


class ImageToTextParser:

    def __init__(self, model: str = ''):
        self.model = model

    @abstractmethod
    def generate(self, image: Path, prompt: str) -> str:
        pass


class DashscopeParser(ImageToTextParser):

    def generate(self, image: Path, prompt: str) -> str:
        import dashscope
        from http import HTTPStatus
        if not os.getenv('DASHSCOPE_API_KEY', None):
            print(
                'Can not parse image to text through dashscope: `DASHSCOPE_API_KEY` is required to be set '
                'to the environment variable. Therefore, only the path and file name information of the '
                'image can be obtained when retrieving. Or you can choose to use other text2image methods.'
            )
            return ''
        """Sample of use local file.
        linux&mac file schema: file:///home/images/test.png
        windows file schema: file://D:/images/abc.png
        """
        local_file_path = f'file://{image.__str__()}'

        messages = [{
            'role': 'user',
            'content': [{
                'image': local_file_path
            }, {
                'text': prompt
            }]
        }]
        response = dashscope.MultiModalConversation.call(
            model=self.model, messages=messages)

        if response.status_code == HTTPStatus.OK:
            return response['output']['choices'][0]['message']['content'][0][
                'text']
        else:  # 如果调用失败
            print(response)
            return ''


class ModelscopeParser(ImageToTextParser):

    def __init__(self,
                 model='iic/cv_convnextTiny_ocr-recognition-document_damo',
                 **kwargs):
        super().__init__(model=model, **kwargs)

        from modelscope.hub.file_download import model_file_download
        from modelscope.hub.utils.utils import get_cache_dir
        from modelscope.pipelines import pipeline
        from modelscope.utils.config import Config
        from modelscope.utils.constant import ModelFile
        cache_root = get_cache_dir()
        configuration_file = os.path.join(cache_root, model,
                                          ModelFile.CONFIGURATION)
        if not os.path.exists(configuration_file):

            configuration_file = model_file_download(
                model_id=model, file_path=ModelFile.CONFIGURATION)
        cfg = Config.from_file(configuration_file)
        task = cfg.safe_get('task')

        self._pipeline = pipeline(task=task, model=model)

    def generate(self, image: Path, **kwargs) -> str:
        res = self._pipeline(image.__str__())
        return res['text'][0]


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
                f'image_parser {image_parser} has not supported yet. Using default image parser dashscope: qwen-vl-max.'
            )
    return DashscopeParser()


class CustomImageReader(BaseReader):

    def __init__(
        self,
        image_parser: Union[Type[ImageToTextParser], ImageToTextParser,
                            None] = None,
        keep_image: bool = False,
        parse_text: bool = True,
        prompt: str = 'Question: describe what you see in this image. Answer:',
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
    fp = 'tests/samples/ms_intro.png'
    a = ModelscopeParser()
    res = a.generate(Path(fp))
    print(res)
