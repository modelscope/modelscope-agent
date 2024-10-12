import os
import shutil
import traceback

import json
from modelscope_agent.tools.utils.openapi_utils import openapi_schema_convert
from modelscope_agent.utils.logger import agent_logger as logger

from modelscope.utils.config import Config

DEFAULT_AGENT_DIR = os.getenv('DEFAULT_AGENT_DIR', '/tmp/agentfabric')
DEFAULT_BUILDER_CONFIG_DIR = os.path.join(DEFAULT_AGENT_DIR, 'config')
DEFAULT_BUILDER_CONFIG_FILE = os.path.join(DEFAULT_BUILDER_CONFIG_DIR,
                                           'builder_config.json')
DEFAULT_OPENAPI_PLUGIN_CONFIG_FILE = os.path.join(
    DEFAULT_BUILDER_CONFIG_DIR, 'openapi_plugin_config.json')
DEFAULT_MODEL_CONFIG_FILE = './config/model_config.json'
DEFAULT_TOOL_CONFIG_FILE = './config/tool_config.json'
DEFAULT_CODE_INTERPRETER_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR',
                                         '/tmp/ci_workspace')
DEFAULT_UUID_HISTORY = os.path.join(DEFAULT_AGENT_DIR, 'history')
DEFAULT_PREVIEW_HISTORY = os.path.join(DEFAULT_AGENT_DIR, 'preview_history')


def get_user_dir(uuid_str=''):
    return os.path.join(DEFAULT_BUILDER_CONFIG_DIR, uuid_str)


def get_ci_dir():
    return DEFAULT_CODE_INTERPRETER_DIR


def get_user_ci_dir(uuid_str='', session_str=''):
    return os.path.join(DEFAULT_CODE_INTERPRETER_DIR, uuid_str, session_str)


def get_user_builder_history_dir(uuid_str='', session_str=''):
    return os.path.join(DEFAULT_UUID_HISTORY, uuid_str, session_str)


def get_user_preview_history_dir(uuid_str='', session_str=''):
    return os.path.join(DEFAULT_UUID_HISTORY, uuid_str, 'preview', session_str)


def get_user_cfg_file(uuid_str=''):
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    # convert from ./config/builder_config.json to ./config/user/builder_config.json
    builder_cfg_file = builder_cfg_file.replace('config/', 'config/user/')

    # convert from ./config/user/builder_config.json to ./config/uuid/builder_config.json
    if uuid_str != '':
        builder_cfg_file = builder_cfg_file.replace('user', uuid_str)
    return builder_cfg_file


def get_user_openapi_plugin_cfg_file(uuid_str=''):
    openapi_plugin_cfg_file = os.getenv('OPENAPI_PLUGIN_CONFIG_FILE',
                                        DEFAULT_OPENAPI_PLUGIN_CONFIG_FILE)
    openapi_plugin_cfg_file = openapi_plugin_cfg_file.replace(
        'config/', 'config/user/')
    if uuid_str != '':
        openapi_plugin_cfg_file = openapi_plugin_cfg_file.replace(
            'user', uuid_str)
    return openapi_plugin_cfg_file


def save_builder_configuration(builder_cfg, uuid_str=''):
    builder_cfg_file = get_user_cfg_file(uuid_str)
    if uuid_str != '' and not os.path.exists(
            os.path.dirname(builder_cfg_file)):
        os.makedirs(os.path.dirname(builder_cfg_file))
    with open(builder_cfg_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(builder_cfg, indent=2, ensure_ascii=False))


def is_valid_plugin_configuration(openapi_plugin_cfg):
    if 'schema' in openapi_plugin_cfg:
        schema = openapi_plugin_cfg['schema']
        if isinstance(schema, dict):
            return True
    else:
        return False


def save_plugin_configuration(openapi_plugin_cfg, uuid_str):
    openapi_plugin_cfg_file = get_user_openapi_plugin_cfg_file(uuid_str)
    if uuid_str != '' and not os.path.exists(
            os.path.dirname(openapi_plugin_cfg_file)):
        os.makedirs(os.path.dirname(openapi_plugin_cfg_file))
    with open(openapi_plugin_cfg_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(openapi_plugin_cfg, indent=2, ensure_ascii=False))


def get_avatar_image(bot_avatar, uuid_str=''):
    user_avatar_path = os.path.join(
        os.path.dirname(__file__), 'assets/user.jpg')
    bot_avatar_path = os.path.join(os.path.dirname(__file__), 'assets/bot.jpg')
    if len(bot_avatar) > 0:
        bot_avatar_path = os.path.join(DEFAULT_BUILDER_CONFIG_DIR, uuid_str,
                                       bot_avatar)
        if uuid_str != '':
            # use default if not exists
            if not os.path.exists(bot_avatar_path):
                # create parents directory
                os.makedirs(os.path.dirname(bot_avatar_path), exist_ok=True)
                # copy the template to the address
                temp_bot_avatar_path = os.path.join(DEFAULT_BUILDER_CONFIG_DIR,
                                                    bot_avatar)
                if not os.path.exists(temp_bot_avatar_path):
                    # fall back to default local avatar image
                    temp_bot_avatar_path = os.path.join('./config', bot_avatar)
                    if not os.path.exists(temp_bot_avatar_path):
                        temp_bot_avatar_path = os.path.join(
                            './config', 'custom_bot_avatar.png')

                shutil.copy(temp_bot_avatar_path, bot_avatar_path)

    return [user_avatar_path, bot_avatar_path]


def save_avatar_image(image_path, uuid_str=''):
    bot_avatar = os.path.basename(image_path)
    bot_avatar_path = os.path.join(DEFAULT_BUILDER_CONFIG_DIR, uuid_str,
                                   bot_avatar)
    shutil.copy(image_path, bot_avatar_path)
    return bot_avatar, bot_avatar_path


def parse_configuration(uuid_str='', use_tool_api=False):
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
        builder_cfg_file_temp = './config/builder_config.json'

        if builder_cfg_file_temp != builder_cfg_file:
            shutil.copy(builder_cfg_file_temp, builder_cfg_file)

    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)

    builder_cfg = Config.from_file(builder_cfg_file)
    model_cfg = Config.from_file(model_cfg_file)
    tool_cfg = Config.from_file(tool_cfg_file)

    tools_info = builder_cfg.tools
    available_tool_list = []
    for key, value in tools_info.items():
        if key in tool_cfg:
            tool_cfg[key]['use'] = value['use']
        else:
            # for tool hub only
            if '/' in key:
                tool_cfg[key] = value
        if value['use']:
            available_tool_list.append(key)

    plugin_cfg = {}
    available_plugin_list = []
    if use_tool_api and getattr(builder_cfg, 'openapi_list', None):
        available_plugin_list = builder_cfg.openapi_list
    else:
        available_plugin_list = []
        openapi_plugin_file = get_user_openapi_plugin_cfg_file(uuid_str)
        openapi_plugin_cfg_file_temp = './config/openapi_plugin_config.json'
        if os.path.exists(openapi_plugin_file):
            openapi_plugin_cfg = Config.from_file(openapi_plugin_file)
            try:
                config_dict = openapi_schema_convert(
                    schema=openapi_plugin_cfg.schema,
                    auth=openapi_plugin_cfg.auth.to_dict())
                plugin_cfg = Config(config_dict)
                for name, config in config_dict.items():
                    available_plugin_list.append(name)
            except Exception as e:
                logger.query_error(
                    uuid=uuid_str,
                    error=str(e),
                    details={
                        'error_traceback':
                        traceback.format_exc(),
                        'error_details':
                        'The format of the plugin config file is incorrect.'
                    })
        elif not os.path.exists(openapi_plugin_file):
            if os.path.exists(openapi_plugin_cfg_file_temp):
                os.makedirs(
                    os.path.dirname(openapi_plugin_file), exist_ok=True)
                if openapi_plugin_cfg_file_temp != openapi_plugin_file:
                    shutil.copy(openapi_plugin_cfg_file_temp,
                                openapi_plugin_file)

    return builder_cfg, model_cfg, tool_cfg, available_tool_list, plugin_cfg, available_plugin_list
