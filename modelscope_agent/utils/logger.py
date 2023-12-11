import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict

import json

# environ params
LOG_FILE_PATH = 'LOG_FILE_PATH'
LOG_MAX_BYTES = 'LOG_MAX_BYTES'
LOG_BACKUP_COUNT = 'LOG_BACKUP_COUNT'
LOG_LEVEL = 'LOG_LEVEL'
LOG_FORMAT = 'LOG_FORMAT'

# constant
LOG_NAME = 'modelscope-agent'
INFO_LOG_FILE_NAME = 'info.log'
ERROR_LOG_FILE_NAME = 'error.log'


def get_root_dir():
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    parent_dir_path = os.path.dirname(current_dir_path)
    grandparent_dir_path = os.path.dirname(parent_dir_path)
    return grandparent_dir_path


class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format.
    """

    def format(self, record):
        log_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage(),
            # Extract additional fields if they are in the 'extra' dict
            'uuid': getattr(record, 'uuid', None),
            'request_id': getattr(record, 'request_id', None),
            'content': getattr(record, 'content', None),
            'step': getattr(record, 'step', None)
        }
        # Clean up any extra fields that are None (not provided)
        log_record = {k: v for k, v in log_record.items() if v is not None}
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class Logger:

    def __init__(self, log_dir, max_bytes=50 * 1024 * 1024, backup_count=7):
        """
        Initialize the Logger with basic configuration.
        Args:
            log_dir (str): Directory to save the log files.
            max_bytes (int): Maximum size in bytes for a single log file, default 50MB.
            backup_count (int): The number of log files to keep, default is 7.
        """
        # Create logger
        self.logger = logging.getLogger(LOG_NAME)
        self.logger.propagate = False
        log_level = os.getenv(LOG_LEVEL, 'INFO').upper()
        self.logger.setLevel(getattr(logging, log_level))

        # Determine the log format
        log_format_env = os.getenv(LOG_FORMAT, 'normal').lower()
        if log_format_env == 'json':
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s')

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create info file handler
        info_file_path = os.path.join(log_dir, INFO_LOG_FILE_NAME)
        info_file_handler = RotatingFileHandler(
            info_file_path,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count)
        info_file_handler.setFormatter(formatter)
        info_file_handler.setLevel(logging.INFO)

        # Create error file handler
        error_file_path = os.path.join(log_dir, ERROR_LOG_FILE_NAME)
        error_file_handler = RotatingFileHandler(
            error_file_path,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count)
        error_file_handler.setFormatter(formatter)
        error_file_handler.setLevel(logging.ERROR)

        # Add handlers to the logger
        self.logger.addHandler(info_file_handler)
        self.logger.addHandler(error_file_handler)

    def get_logger(self):
        """
        Returns the configured logger object.
        Returns:
            logging.Logger: The configured logger object.
        """
        return self.logger


class AgentLogger:

    def __init__(self):
        self._init_loger()

    def _init_loger(self):
        # Initialize Logger with the path to the log file
        _log_dir = os.getenv(LOG_FILE_PATH, f'{get_root_dir()}/logs')
        os.makedirs(_log_dir, exist_ok=True)
        self.logger = Logger(
            log_dir=_log_dir,
            max_bytes=int(os.environ.get(LOG_MAX_BYTES, 50 * 1024 * 1024)),
            backup_count=int(os.environ.get(LOG_BACKUP_COUNT,
                                            7))).get_logger()

    def info(self,
             uuid: str = 'default_user',
             request_id: str = 'default_request_id',
             content: Dict = None,
             step: str = '',
             message: str = ''):
        if content is None:
            content = {}

        self.logger.info(
            message,
            extra={
                'uuid': uuid,
                'request_id': request_id,
                'content': content,
                'step': step
            })

    def error(self,
              uuid: str = 'default_user',
              request_id: str = 'default_request_id',
              content: Dict = None,
              step: str = '',
              message: str = ''):
        if content is None:
            content = {}

        self.logger.error(
            message,
            extra={
                'uuid': uuid,
                'request_id': request_id,
                'content': content,
                'step': step
            })


logger = AgentLogger()
