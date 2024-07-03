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
        api_key: str = os.getenv('DASHSCOPE_API_KEY', '')):
        from openai import OpenAI

        assert len(api_key), 'api_key is not set.'
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        super().__init__(model)

    def generate(self, image: Path, prompt: str) -> str:
        import base64
        import mimetypes

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

                completion = self.client.chat.completions.create(
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
    a = OpenaiAPIParser()
    res = a.generate(Path(fp), '图片的内容是什么？')
    print(res)
