import os
from urllib.parse import unquote_plus, urlparse

from dashscope.common.error import InvalidInput, UploadFileException
from dashscope.utils.oss_utils import OssUtils

FILE_PATH_SCHEMA = 'file://'


def get_upload_url(model: str, file_to_upload: str, api_key: str):
    """This function is used to convert local file to get its oss url.

    Args:
        model(str): Theoretically, you can set this parameter freely. It will only affect
                    the information of the oss url and will not affect the function function.
        file_to_upload(str): the local file path which you need to convert to oss url.And it should
                            start with 'file://'.
        api_key(str): dashscope_api_key which you have set in enviroment.

    Returns:
        An oss type url.

    Raises:
        InvalidInput: the file path you upload is not exists.
    """
    if file_to_upload.startswith(FILE_PATH_SCHEMA):
        parse_result = urlparse(file_to_upload)
        if parse_result.netloc:
            file_path = parse_result.netloc + unquote_plus(parse_result.path)
        else:
            file_path = unquote_plus(parse_result.path)
        if os.path.exists(file_path):
            file_url = OssUtils.upload(
                model=model, file_path=file_path, api_key=api_key)
            if file_url is None:
                raise UploadFileException('Uploading file: %s failed'
                                          % file_to_upload)
            return file_url
        else:
            raise InvalidInput('The file: %s is not exists!' % file_path)
    return None
