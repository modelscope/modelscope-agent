import os
import shutil

import json

from modelscope.utils.config import Config

DEFAULT_BUILDER_CONFIG_DIR = './config'
DEFAULT_BUILDER_CONFIG_FILE = './config/builder_config.json'
DEFAULT_MODEL_CONFIG_FILE = './config/model_config.json'
DEFAULT_TOOL_CONFIG_FILE = './config/tool_config.json'


def save_builder_configuration(builder_cfg):
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    with open(builder_cfg_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(builder_cfg, indent=2, ensure_ascii=False))


def get_avatar_image(bot_avatar):
    user_avatar_path = os.path.join(
        os.path.dirname(__file__), 'assets/user.jpg')
    bot_avatar_path = os.path.join(os.path.dirname(__file__), 'assets/bot.jpg')
    if len(bot_avatar) > 0:
        bot_avatar_path = os.path.join(
            os.path.dirname(__file__), DEFAULT_BUILDER_CONFIG_DIR, bot_avatar)
    return [user_avatar_path, bot_avatar_path]


def save_avatar_image(image_path):
    file_extension = os.path.splitext(image_path)[1]
    bot_avatar = f'custom_bot_avatar{file_extension}'
    bot_avatar_path = os.path.join(
        os.path.dirname(__file__), DEFAULT_BUILDER_CONFIG_DIR, bot_avatar)
    shutil.copy(image_path, bot_avatar_path)
    return bot_avatar, bot_avatar_path


def parse_configuration():
    """parse configuration

    Args:

    Returns:
        dict: parsed configuration

    """
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE', DEFAULT_MODEL_CONFIG_FILE)
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
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
