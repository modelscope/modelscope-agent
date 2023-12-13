import os

from modelscope_agent.utils.logger import logger

LOG_FILE_PATH = 'LOG_FILE_PATH'
LOG_MAX_BYTES = 'LOG_MAX_BYTES'
LOG_BACKUP_COUNT = 'LOG_BACKUP_COUNT'


def get_agentfabric_logger():

    _log_dir = os.getenv(LOG_FILE_PATH, f'{os.getcwd()}/logs')
    os.makedirs(_log_dir, exist_ok=True)
    logger.set_file_handle(
        log_dir=_log_dir,
        max_bytes=int(os.environ.get(LOG_MAX_BYTES, 50 * 1024 * 1024)),
        backup_count=int(os.environ.get(LOG_BACKUP_COUNT, 7)))
    return logger
