import os
from configparser import ConfigParser

import oss2

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


def upload_to_oss(bucket, local_file_path, oss_file_path):
    # 上传文件到阿里云OSS
    bucket.put_object_from_file(oss_file_path, local_file_path)

    # 设置文件的公共读权限
    bucket.put_object_acl(oss_file_path, oss2.OBJECT_ACL_PUBLIC_READ)

    # 获取文件的公共链接
    file_url = f"https://{bucket.bucket_name}.{bucket.endpoint.replace('http://', '')}/{oss_file_path}"
    return file_url


def get_oss_config():
    # 尝试从环境变量中读取配置
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
    access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
    endpoint = os.getenv('OSS_ENDPOINT')
    bucket_name = os.getenv('OSS_BUCKET_NAME')

    # 如果环境变量没有设置，尝试从.ossutilconfig文件中读取
    if not access_key_id or not access_key_secret or not endpoint or not bucket_name:
        config = ConfigParser()
        config.read(os.path.expanduser('~/.ossutilconfig'))
        if 'Credentials' in config:
            access_key_id = config.get('Credentials', 'accessKeyId')
            access_key_secret = config.get('Credentials', 'accessKeySecret')
            endpoint = config.get('Credentials', 'endpoint')
            bucket_name = config.get('Credentials', 'bucketName')

    return access_key_id, access_key_secret, endpoint, bucket_name


def path2url(local_file_path, oss_file_path):
    local_file_path = os.path.join(WORK_DIR, local_file_path)
    ak_id, ak_secret, endpoint, bucket_name = get_oss_config()
    auth = oss2.Auth(ak_id, ak_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    file_url = upload_to_oss(bucket, local_file_path,
                             f'agents/user/{oss_file_path}')
    return file_url
