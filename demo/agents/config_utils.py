import json
import os
from demo.agents.builder_core import DEFAULT_BUILDER_CONFIG_FILE, DEFAULT_MODEL_CONFIG_FILE, DEFAULT_TOOL_CONFIG_FILE


def save_builder_configuration(builder_cfg):
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    with open(builder_cfg_file, 'w') as f:
        f.write(json.dumps(builder_cfg, indent=2))


def load_assets_configuration():
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE', DEFAULT_MODEL_CONFIG_FILE)
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)
    with open(model_cfg_file, 'r') as file:
        model_cfg = json.load(file)
    with open(tool_cfg_file, 'r') as file:
        tool_cfg = json.load(file)

    return model_cfg, tool_cfg
