import os

import json

is_cn_env = os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio'

api_key = os.getenv('MODELSCOPE_API_KEY')

internal_mcp_config = json.loads(
    os.getenv('INTERNAL_MCP_CONFIG', '{"mcpServers": {}}'))

# oss
endpoint = os.getenv('OSS_ENDPOINT')

region = os.getenv('OSS_REGION')

bucket_name = os.getenv('OSS_BUCKET_NAME')
