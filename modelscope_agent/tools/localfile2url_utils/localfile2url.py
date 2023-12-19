import os
from urllib.parse import unquote_plus, urlparse

from dashscope.common.error import InvalidInput, UploadFileException
from dashscope.utils.oss_utils import OssUtils

FILE_PATH_SCHEMA = 'file://'


def get_upload_url(model: str, upload_path: str, api_key: str):
    if upload_path.startswith(FILE_PATH_SCHEMA):
        parse_result = urlparse(upload_path)
        if parse_result.netloc:
            file_path = parse_result.netloc + unquote_plus(parse_result.path)
        else:
            file_path = unquote_plus(parse_result.path)
        if os.path.exists(file_path):
            file_url = OssUtils.upload(
                model=model, file_path=file_path, api_key=api_key)
            if file_url is None:
                raise UploadFileException('Uploading file: %s failed'
                                          % upload_path)
            return file_url
        else:
            raise InvalidInput('The file: %s is not exists!' % file_path)
    return None
