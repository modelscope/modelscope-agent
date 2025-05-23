import uuid
from http import HTTPStatus

import oss2
from env import bucket_name, endpoint, region
from oss2.credentials import EnvironmentVariableCredentialsProvider

# OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET。
auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())


def file_path_to_oss_url(file_path: str):
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    if file_path.startswith('http'):
        return file_path
    ext = file_path.split('.')[-1]
    object_name = f'studio-temp/mcp-playground/{uuid.uuid4()}.{ext}'
    response = bucket.put_object_from_file(object_name, file_path)
    file_url = file_path
    if response.status == HTTPStatus.OK:
        file_url = bucket.sign_url(
            'GET', object_name, 60 * 60, slash_safe=True)
    return file_url
