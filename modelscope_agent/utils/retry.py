import time
from functools import wraps
from traceback import format_exc

from modelscope_agent.utils.logger import agent_logger as logger


def retry(max_retries=3, delay_seconds=1, return_str=False):
    """
    Retry decorator with exponential backoff.
    Args:
        max_retries: max retry times
        delay_seconds: delay seconds between retries
        return_str: want to return in str format, set it to True

    Returns:func

    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    raise AssertionError(e)
                except Exception:
                    logger.warning(
                        f'Attempt to run {func.__name__} {attempts + 1} failed: {format_exc()}'
                    )
                    attempts += 1
                    time.sleep(delay_seconds)
            if return_str:
                return f'Max retries reached. Attempt to run {func.__name__} failed after {max_retries} times'
            else:
                raise Exception('Max retries reached. Failed to get result')

        return wrapper

    return decorator
