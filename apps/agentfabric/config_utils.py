import os
import shutil
from typing import Dict

import json

from modelscope.utils.config import Config

DEFAULT_BUILDER_CONFIG_DIR = './config'
DEFAULT_BUILDER_CONFIG_FILE = './config/builder_config.json'
DEFAULT_MODEL_CONFIG_FILE = './config/model_config.json'
DEFAULT_TOOL_CONFIG_FILE = './config/tool_config.json'


def get_user_cfg_file(uuid_str=''):
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    # convert from ./config/builder_config.json to ./config/user/builder_config.json
    builder_cfg_file = builder_cfg_file.replace('config/', 'config/user/')

    # convert from ./config/user/builder_config.json to ./config/uuid/builder_config.json
    if uuid_str != '':
        builder_cfg_file = builder_cfg_file.replace('user', uuid_str)
    return builder_cfg_file


def save_builder_configuration(builder_cfg, uuid_str=''):
    builder_cfg_file = get_user_cfg_file(uuid_str)
    if uuid_str != '' and not os.path.exists(
            os.path.dirname(builder_cfg_file)):
        os.makedirs(os.path.dirname(builder_cfg_file))
    with open(builder_cfg_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(builder_cfg, indent=2, ensure_ascii=False))


def get_avatar_image(bot_avatar, uuid_str=''):
    user_avatar_path = os.path.join(
        os.path.dirname(__file__), 'assets/user.jpg')
    bot_avatar_path = os.path.join(os.path.dirname(__file__), 'assets/bot.jpg')
    if len(bot_avatar) > 0:
        bot_avatar_path = os.path.join(
            os.path.dirname(__file__), DEFAULT_BUILDER_CONFIG_DIR, 'user',
            bot_avatar)
        if uuid_str != '':
            bot_avatar_path = bot_avatar_path.replace('user', uuid_str)
            # use default if not exists
            if not os.path.exists(bot_avatar_path):
                # create parents directory
                os.makedirs(os.path.dirname(bot_avatar_path), exist_ok=True)
                # copy the template to the address
                temp_bot_avatar_path = os.path.join(
                    os.path.dirname(__file__), DEFAULT_BUILDER_CONFIG_DIR,
                    bot_avatar)

                shutil.copy(temp_bot_avatar_path, bot_avatar_path)

    return [user_avatar_path, bot_avatar_path]


def save_avatar_image(image_path, uuid_str=''):
    bot_avatar = os.path.basename(image_path)
    bot_avatar_path = os.path.join(
        os.path.dirname(__file__), DEFAULT_BUILDER_CONFIG_DIR, 'user',
        bot_avatar)
    bot_avatar_path = bot_avatar_path.replace('user', uuid_str)
    shutil.copy(image_path, bot_avatar_path)
    return bot_avatar, bot_avatar_path


def parse_configuration(uuid_str=''):
    """parse configuration

    Args:

    Returns:
        dict: parsed configuration

    """
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE', DEFAULT_MODEL_CONFIG_FILE)

    builder_cfg_file = get_user_cfg_file(uuid_str)
    # use default if not exists
    if not os.path.exists(builder_cfg_file):
        # create parents directory
        os.makedirs(os.path.dirname(builder_cfg_file), exist_ok=True)
        # copy the template to the address
        builder_cfg_file_temp = os.environ.get('BUILDER_CONFIG_FILE',
                                               DEFAULT_BUILDER_CONFIG_FILE)
        if builder_cfg_file_temp != builder_cfg_file:
            shutil.copy(builder_cfg_file_temp, builder_cfg_file)

    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)

    builder_cfg = Config.from_file(builder_cfg_file)
    model_cfg = Config.from_file(model_cfg_file)
    tool_cfg = Config.from_file(tool_cfg_file)

    tools_info = builder_cfg.tools
    available_tool_list = []
    for key, value in tools_info.items():
        if value['use']:
            available_tool_list.append(key)

    return builder_cfg, model_cfg, tool_cfg, available_tool_list
