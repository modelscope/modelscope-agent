import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict

import json

# environ params
LOG_LEVEL = 'LOG_LEVEL'
LOG_CONSOLE_FORMAT = 'LOG_CONSOLE_FORMAT'
LOG_FILE_FORMAT = 'LOG_FILE_FORMAT'
LOG_ENABLE_FILE = 'LOG_ENABLE_FILE'
LOG_FILE_PATH = 'LOG_FILE_PATH'
LOG_MAX_BYTES = 'LOG_MAX_BYTES'
LOG_BACKUP_COUNT = 'LOG_BACKUP_COUNT'

# constant
LOG_NAME = 'modelscope-agent'
INFO_LOG_FILE_NAME = 'info.log'
ERROR_LOG_FILE_NAME = 'error.log'


def get_formatter(log_format_env):
    if log_format_env == 'json':
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter()
    return formatter


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
            # 'request_id': getattr(record, 'request_id', None),
            'details': getattr(record, 'details', None),
            'error': getattr(record, 'error', None),
            'step': getattr(record, 'step', None)
        }
        # Clean up any extra fields that are None (not provided)
        log_record = {k: v for k, v in log_record.items() if v is not None}
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """
    Custom formatter to output logs in text format.
    """

    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        message = record.getMessage()

        # Collect additional fields if they are in the 'extra' dict
        uuid = getattr(record, 'uuid', '-')
        # request_id = getattr(record, 'request_id', '-')
        details = getattr(record, 'details', '-')
        step = getattr(record, 'step', '-')
        error = getattr(record, 'error', '-')

        # Format the log record as text
        log_message = f'{timestamp} - {LOG_NAME} - {level} - '
        log_message = log_message + f' | message: {message}'
        if uuid != '-':
            log_message += f' | uuid: {uuid}'
        # if request_id != '-':
        #     log_message += f' | request_id: {request_id}'
        if details != '-':
            log_message += f' | details: {details}'
        if step != '-':
            log_message += f' | step: {step}'
        if error != '-':
            log_message += f' | error: {error}'

        if record.exc_info:
            log_message += f' | Exception: {self.formatException(record.exc_info)}'

        return log_message


class AgentLogger:
    r"""
    The AgentLogger class has two modes of operation: one is for global logging,
    which allows the use of functions such as info; the other is for query-level
    logging, which requires the use of a UUID in conjunction with functions like
    query_info.

    Examples:
    ```python
    >>> agent_logger = AgentLogger()
    >>> agent_logger.info('simple log')

    >>> agent_logger.query_info(
    >>>     uuid=uuid_str,
    >>>     details={
    >>>         'info': 'complex log'
    >>>     }
    >>> )
    """

    def __init__(self):
        self.logger = logging.getLogger(LOG_NAME)
        self.logger.propagate = False
        log_level = os.getenv(LOG_LEVEL, 'INFO').upper()
        self.logger.setLevel(getattr(logging, log_level))

        # Create console handler with TextFormatter
        console_log_formatter = get_formatter(
            os.getenv(LOG_CONSOLE_FORMAT, 'normal').lower())
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            get_formatter(log_format_env=console_log_formatter))
        self.logger.addHandler(console_handler)

        if os.environ.get(LOG_ENABLE_FILE, 'on').lower() == 'on':
            _log_dir = os.getenv(LOG_FILE_PATH, f'{os.getcwd()}/logs')
            os.makedirs(_log_dir, exist_ok=True)
            self.set_file_handle(
                log_dir=_log_dir,
                max_bytes=int(os.environ.get(LOG_MAX_BYTES, 50 * 1024 * 1024)),
                backup_count=int(os.environ.get(LOG_BACKUP_COUNT, 7)))

    def set_file_handle(self,
                        log_dir,
                        max_bytes=50 * 1024 * 1024,
                        backup_count=7):
        """
        Args:
            log_dir (str): Directory to save the log files.
            max_bytes (int): Maximum size in bytes for a single log file, default 50MB.
            backup_count (int): The number of log files to keep, default is 7.
        """
        # Create file handlers with JsonFormatter
        file_log_formatter = get_formatter(
            os.getenv(LOG_FILE_FORMAT, 'json').lower())
        info_file_path = os.path.join(log_dir, INFO_LOG_FILE_NAME)
        info_file_handler = RotatingFileHandler(
            info_file_path,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count)
        info_file_handler.setFormatter(file_log_formatter)
        info_file_handler.setLevel(logging.INFO)

        # Create error file handler
        error_file_path = os.path.join(log_dir, ERROR_LOG_FILE_NAME)
        error_file_handler = RotatingFileHandler(
            error_file_path,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count)
        error_file_handler.setFormatter(file_log_formatter)
        error_file_handler.setLevel(logging.ERROR)

        # Add handlers to the logger
        self.logger.addHandler(info_file_handler)
        self.logger.addHandler(error_file_handler)

    def info(self, message: str, *args):
        self.logger.info(message, *args)

    def query_info(
            self,
            uuid: str = 'default_user',
            # request_id: str = 'default_request_id',
            details: Dict = None,
            step: str = '',
            message: str = ''):
        if details is None:
            details = {}

        self.logger.info(
            message,
            extra={
                'uuid': uuid,
                # 'request_id': request_id,
                'details': details,
                'step': step,
                'error': ''
            })

    def error(self, message: str = '', *args):
        self.logger.error(message, *args)

    def query_error(
            self,
            uuid: str = 'default_user',
            # request_id: str = 'default_request_id',
            details: Dict = None,
            step: str = '',
            message: str = '',
            error: str = ''):
        if details is None:
            details = {}

        self.logger.error(
            message,
            extra={
                'uuid': uuid,
                # 'request_id': request_id,
                'details': details,
                'step': step,
                'error': error
            })

    def warning(self, message: str = '', *args):
        self.logger.warning(message, *args)

    def query_warning(
            self,
            uuid: str = 'default_user',
            # request_id: str = 'default_request_id',
            details: Dict = None,
            step: str = '',
            message: str = ''):
        if details is None:
            details = {}

        self.logger.warning(
            message,
            extra={
                'uuid': uuid,
                # 'request_id': request_id,
                'details': details,
                'step': step,
                'error': ''
            })


agent_logger = AgentLogger()
