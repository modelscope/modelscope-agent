import os
import time

import oss2


class OssStorage(object):

    def __init__(self):
        oss_access_key_id = os.getenv('OSS_ACCESS_KEY_ID', None)
        oss_access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET', None)
        oss_bucket = os.getenv('OSS_BUCKET_NAME', None)
        oss_endpoint = os.getenv('OSS_ENDPOINT', None)
        if not oss_access_key_id or not oss_access_key_secret or not oss_bucket or not oss_endpoint:
            raise ValueError(
                'OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET_NAME, OSS_ENDPOINT must be set'
            )
        self.auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
        self.bucket = oss2.Bucket(self.auth, oss_endpoint, oss_bucket)
        self.endpoint = oss_endpoint
        self.bucket_name = oss_bucket

    def upload(self,
               src_file,
               oss_path,
               max_retries=3,
               retry_delay=1,
               delete_src=True):
        for i in range(max_retries):
            try:
                with open(src_file, 'rb') as f:
                    print(f'src address is {src_file}')
                    modality_data = f.read()
                    result = self.bucket.put_object(oss_path, modality_data)
                    print(result)
                break
            except Exception as e:
                print(f'Error uploading file: {e}')
                if i < max_retries - 1:
                    print(f'Retrying in {retry_delay} seconds...')
                    time.sleep(retry_delay)
                else:
                    os.remove(src_file)
                    raise IOError(f'Exceed the Max retry with error {e}')

        if delete_src:
            os.remove(src_file)

    def uploads(self,
                src_files,
                oss_paths,
                max_retries=3,
                retry_delay=1,
                delete_src=True):
        # get a list of files
        for idx, src_file in enumerate(src_files):
            oss_path = oss_paths[idx]
            self.upload(src_file, oss_path, max_retries, retry_delay,
                        delete_src)

    def get(self, oss_path):
        return self.bucket.get_object(oss_path)

    def get_signed_url(self, oss_path, expire_seconds=3 * 24 * 60 * 60):
        url = self.bucket.sign_url(
            'GET', oss_path, expire_seconds, slash_safe=True)
        return url


if __name__ == '__main__':
    from PIL import Image
    from io import BytesIO

    oss = OssStorage()
    oss_path_test = 'zzc/test.png'
    oss.upload('/Users/zhicheng/zzc2.png', oss_path_test, delete_src=False)
    result1 = oss.get(oss_path_test)
    image_data = Image.open(BytesIO(result1.read()))
    image_data.show()
