from enum import Enum
from pathlib import Path

DEFAULT_AGENT_ROOT = Path.home() / '.modelscope-agent'
DEFAULT_LOG_STORAGE_PATH = DEFAULT_AGENT_ROOT / 'log'
DEFAULT_SEND_TO = 'all'
USER_REQUIREMENT = 'user_requirement'
ENVIRONMENT_NAME = 'env'
AGENT_REGISTRY_NAME = 'agent_center'
TASK_CENTER_NAME = 'task_center'
DEFAULT_TOOL_MANAGER_SERVICE_URL = 'http://localhost:31511'
DEFAULT_ASSISTANT_SERVICE_URL = 'http://localhost:31512'
MODELSCOPE_AGENT_TOKEN_HEADER_NAME = 'X-Modelscope-Agent-Token'
DEFAULT_CODE_INTERPRETER_DIR = '/tmp/ci_workspace'
LOCAL_FILE_PATHS = 'local_file_paths'
BASE64_FILES = 'base64_files'


class ApiNames(Enum):
    dashscope_api_key = 'DASHSCOPE_API_KEY'
    modelscope_api_key = 'MODELSCOPE_API_TOKEN'
    amap_api_key = 'AMAP_TOKEN'
    bing_api_key = 'BING_SEARCH_V7_SUBSCRIPTION_KEY'
    zhipu_api_key = 'ZHIPU_API_KEY'
