from pathlib import Path

DEFAULT_AGENT_ROOT = Path.home() / '.modelscope-agent'
DEFAULT_LOG_STORAGE_PATH = DEFAULT_AGENT_ROOT / 'log'
DEFAULT_SEND_TO = 'all'
USER_REQUIREMENT = 'user_requirement'
ENVIRONMENT_NAME = 'env'
AGENT_REGISTRY_NAME = 'agent_center'
TASK_CENTER_NAME = 'task_center'
DEFAULT_TOOL_MANAGER_SERVICE_URL = 'http://localhost:31511'
