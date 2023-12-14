import glob
import os
import re
import shutil
import zipfile
from configparser import ConfigParser
from urllib.parse import unquote, urlparse

import json
import oss2
import requests

from modelscope.utils.config import Config

MS_VERSION = '0.2.2rc1'
DEFAULT_MS_PKG = 'https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/releases/v/modelscope_agent-version-py3-none-any.whl'  # noqa E501


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


def pop_user_info_from_config(src_dir, uuid_str):
    """ Remove all personal information from the configuration files and return this data.
    The purpose of this is to ensure that personal information is not stored in plain text
    when releasing.

    Args:
        src_dir (str): config root path
        uuid_str (str): user id
    """
    user_info = {}

    # deal with plugin cfg
    plugin_config_path = f'{src_dir}/config/{uuid_str}/openapi_plugin_config.json'
    if os.path.exists(plugin_config_path):
        with open(plugin_config_path, 'r') as f:
            plugin_config = json.load(f)
        if 'auth' in plugin_config:
            if plugin_config['auth']['type'] == 'API Key':
                user_info['apikey'] = plugin_config['auth'].pop('apikey')
                user_info['apikey_type'] = plugin_config['auth'].pop(
                    'apikey_type')
        with open(plugin_config_path, 'w') as f:
            json.dump(plugin_config, f, indent=2, ensure_ascii=False)

    return user_info


def prepare_agent_zip(agent_name, src_dir, uuid_str, state):
    # 设置阿里云OSS的认证信息
    local_file = os.path.abspath(os.path.dirname(__file__))
    ak_id, ak_secret, endpoint, bucket_name = get_oss_config()
    auth = oss2.Auth(ak_id, ak_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    new_directory = f'{src_dir}/upload/{uuid_str}'  # 新目录的路径

    # 创建新目录
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
        os.makedirs(new_directory)

    # 复制config下的uuid_str目录到new_directory下并改名为local_user
    uuid_str_path = f'{src_dir}/config/{uuid_str}'  # 指向uuid_str目录的路径
    local_user_path = f'{new_directory}/config'  # 新的目录路径
    shutil.copytree(uuid_str_path, local_user_path, dirs_exist_ok=True)

    target_conf = os.path.join(local_user_path, 'builder_config.json')
    builder_cfg = Config.from_file(target_conf)
    builder_cfg.knowledge = [
        'config/' + f.split('/')[-1] for f in builder_cfg.knowledge
    ]
    with open(target_conf, 'w') as f:
        json.dump(builder_cfg.to_dict(), f, indent=2, ensure_ascii=False)

    # 复制config目录下所有.json文件到new_directory/config
    config_path = f'{local_file}/config'
    new_config_path = f'{new_directory}/config'

    def find_json_and_images(directory):
        # 确保路径以斜杠结束
        directory = os.path.join(directory, '')

        # 找到所有的JSON文件
        json_files = [
            os.path.join(directory, 'model_config.json'),
            os.path.join(directory, 'tool_config.json'),
        ]

        # 找到所有的图片文件
        image_files = glob.glob(directory + '*.png') + \
            glob.glob(directory + '*.jpg') + \
            glob.glob(directory + '*.jpeg') + \
            glob.glob(directory + '*.gif')  # 根据需要可以添加更多图片格式

        return json_files + image_files

    for f in find_json_and_images(config_path):
        shutil.copy(f, new_config_path)

    # 复制assets目录到new_directory
    assets_path = f'{local_file}/assets'
    new_assets_path = f'{new_directory}/assets'
    shutil.copytree(assets_path, new_assets_path, dirs_exist_ok=True)

    # 在requirements.txt中添加新的行
    requirements_file = f'{local_file}/requirements.txt'
    new_requirements_file = f'{new_directory}/requirements.txt'
    modelscope_agent_pkg = DEFAULT_MS_PKG.replace('version', MS_VERSION)
    with open(requirements_file, 'r') as file:
        content = file.readlines()
    with open(new_requirements_file, 'w') as file:
        file.write(modelscope_agent_pkg + '\n')
        file.writelines(content)

    # 复制.py文件到新目录
    for file in os.listdir(local_file):
        if file.endswith('.py'):
            shutil.copy(f'{local_file}/{file}', new_directory)

    # 打包新目录
    archive_path = shutil.make_archive(new_directory, 'zip', new_directory)

    # 使用抽象出的函数上传到OSS并设置权限
    file_url = upload_to_oss(bucket, archive_path,
                             f'agents/user/{uuid_str}/{agent_name}.zip')

    shutil.rmtree(new_directory)

    # 获取必须设置的envs
    envs_required = {}
    for t in builder_cfg.tools:
        if t == 'amap_weather':
            envs_required['AMAP_TOKEN'] = 'Your-AMAP-TOKEN'
    return file_url, envs_required


def parse_version_from_file(file_path):
    # 用于匹配 __version__ 行的正则表达式
    version_pattern = r"^__version__\s*=\s*['\"]([^'\"]+)['\"]"

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 检查每一行是否匹配版本模式
                match = re.match(version_pattern, line.strip())
                if match:
                    # 返回匹配的版本号
                    return match.group(1)
        return None  # 如果文件中没有找到版本号
    except FileNotFoundError:
        return None  # 如果文件不存在


def reload_agent_zip(agent_url, dst_dir, uuid_str, state):
    # download zip from agent_url, and unzip to dst_dir/uuid_str
    # 从URL中解析出文件名
    parsed_url = urlparse(agent_url)
    filename = os.path.basename(parsed_url.path)
    zip_path = os.path.join(dst_dir, filename)

    # 提取agent_name（去掉'.zip'）
    agent_name, _ = os.path.splitext(filename)

    # 创建临时解压目录
    temp_extract_dir = os.path.join(dst_dir, f'temp_{uuid_str}')
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir)

    # 下载ZIP文件
    response = requests.get(agent_url)
    if response.status_code == 200:
        with open(zip_path, 'wb') as file:
            file.write(response.content)
    else:
        raise RuntimeError(
            f'download file from {agent_url} error:\n {response.reason}')

    # 解压ZIP文件到临时目录
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # 解析version信息
    version = parse_version_from_file(
        os.path.join(temp_extract_dir, '/version.py'))
    print(f'agent fabric version: {version}')
    # 创建目标config路径
    target_config_path = os.path.join(dst_dir, 'config', uuid_str)
    if os.path.exists(target_config_path):
        shutil.rmtree(target_config_path)
    os.makedirs(target_config_path)

    # 复制config目录
    # 兼容老版本配置放到local_user目录下，以及新版本直接放在config目录下
    if os.path.exists(os.path.join(temp_extract_dir, 'config', 'local_user')):
        config_source_path = os.path.join(temp_extract_dir, 'config',
                                          'local_user')
    elif os.path.exists(os.path.join(temp_extract_dir, 'config')):
        config_source_path = os.path.join(temp_extract_dir, 'config')
    else:
        raise RuntimeError('未找到正确的配置文件信息')

    if os.path.exists(config_source_path):
        for item in os.listdir(config_source_path):
            s = os.path.join(config_source_path, item)
            d = os.path.join(target_config_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # 清理：删除临时目录和下载的ZIP文件
    shutil.rmtree(temp_extract_dir)
    os.remove(zip_path)

    # 修改知识库路径 config/xxx  to /tmp/agentfabric/config/$uuid/xxx
    target_conf = os.path.join(target_config_path, 'builder_config.json')
    builder_cfg = Config.from_file(target_conf)
    builder_cfg.knowledge = [
        f'{target_config_path}/' + f.split('/')[-1]
        for f in builder_cfg.knowledge
    ]
    with open(target_conf, 'w') as f:
        json.dump(builder_cfg.to_dict(), f, indent=2, ensure_ascii=False)

    return agent_name


if __name__ == '__main__':
    src_dir = os.path.abspath(os.path.dirname(__file__))
    url, envs = prepare_agent_zip('test', src_dir, 'local_user', {})
    print(url)

    agent_name = reload_agent_zip(url, '/tmp/agentfabric_test', 'local_user',
                                  {})
    print(agent_name)
