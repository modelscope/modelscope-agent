import os
import shutil
import oss2
from configparser import ConfigParser
import glob

def upload_to_oss(bucket, local_file_path, oss_file_path):
    # 上传文件到阿里云OSS
    bucket.put_object_from_file(oss_file_path, local_file_path)

    # 设置文件的公共读权限
    bucket.put_object_acl(oss_file_path, oss2.OBJECT_ACL_PUBLIC_READ)

    # 获取文件的URL
    file_url = bucket.sign_url('GET', oss_file_path, 60 * 60)  # 签名URL有效期为1小时
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


def prepare_agent_zip(agent_name, uuid_str, state):
    # 设置阿里云OSS的认证信息
    ak_id, ak_secret, endpoint, bucket_name = get_oss_config()
    auth = oss2.Auth(ak_id, ak_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    src_dir = os.path.abspath(os.path.dirname(__file__))
    new_directory = f'{src_dir}/upload/{uuid_str}'  # 新目录的路径

    # 创建新目录
    if os.path.exists(new_directory):
        os.removedirs(new_directory)
        os.makedirs(new_directory)
    
    # 复制config下的uuid_str目录到new_directory下并改名为local_user
    uuid_str_path = f'{src_dir}/config/{uuid_str}'  # 指向uuid_str目录的路径
    local_user_path = f'{new_directory}/config/local_user'  # 新的目录路径
    shutil.copytree(uuid_str_path, local_user_path, dirs_exist_ok=True)

    # 复制config目录下所有.json文件到new_directory/config
    config_path = f'{src_dir}/config'
    new_config_path = f'{new_directory}/config'

    def find_json_and_images(directory):
        # 确保路径以斜杠结束
        directory = os.path.join(directory, '')

        # 找到所有的JSON文件
        json_files = glob.glob(directory + '*.json')

        # 找到所有的图片文件
        image_files = glob.glob(directory + '*.png') + \
                    glob.glob(directory + '*.jpg') + \
                    glob.glob(directory + '*.jpeg') + \
                    glob.glob(directory + '*.gif')  # 根据需要可以添加更多图片格式

        return json_files + image_files

    for f in find_json_and_images(config_path):
        shutil.copy(f, new_config_path)

    # 复制assets目录到new_directory
    assets_path = f'{src_dir}/assets'
    new_assets_path = f'{new_directory}/assets'
    shutil.copytree(assets_path, new_assets_path, dirs_exist_ok=True)

    # 在requirements.txt中添加新的行
    requirements_file = f'{src_dir}/requirements.txt'
    new_requirements_file = f'{new_directory}/requirements.txt'
    with open(requirements_file, 'r') as file:
        content = file.readlines()
    with open(new_requirements_file, 'w') as file:
        file.write('git+https://github.com/modelscope/modelscope-agent.git\n')
        file.writelines(content)

    # 复制.py文件到新目录
    for file in os.listdir(src_dir):
        if file.endswith('.py'):
            shutil.copy(f'{src_dir}/{file}', new_directory)

    # 打包新目录
    archive_path = shutil.make_archive(new_directory, 'zip', new_directory)

    # 使用抽象出的函数上传到OSS并设置权限
    file_url = upload_to_oss(bucket, archive_path, f'agents/user/{uuid_str}/{agent_name}.zip')

    shutil.rmtree(new_directory)

    return file_url

if __name__ == '__main__':
    url = prepare_agent_zip('test', 'local_user', {})
    print(url)