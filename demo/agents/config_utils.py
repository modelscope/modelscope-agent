import json
import os
from builder_core import DEFAULT_BUILDER_CONFIG_FILE


def save_builder_configuration(builder_cfg):
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    with open(builder_cfg_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(builder_cfg, indent=2, ensure_ascii=False))
