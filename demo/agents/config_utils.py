import os

import json

from modelscope.utils.config import Config

DEFAULT_BUILDER_CONFIG_FILE = "./config/builder_config.json"
DEFAULT_MODEL_CONFIG_FILE = "./config/model_config.json"
DEFAULT_TOOL_CONFIG_FILE = "./config/tool_config.json"


def save_builder_configuration(builder_cfg):
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    with open(builder_cfg_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(builder_cfg, indent=2, ensure_ascii=False))


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
