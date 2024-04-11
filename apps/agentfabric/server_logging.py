import logging
import os
from contextvars import ContextVar

request_id_var = ContextVar('request_id', default='')


# 创建一个日志过滤器，用于加入request_id到日志记录中
class RequestIDLogFilter(logging.Filter):

    def filter(self, record):
        record.request_id = request_id_var.get('')
        record.ip_addr = os.getenv('ALIYUN_ECI_ETH0_IP', '')
        return True


# 设置日志格式
formatter = logging.Formatter(
    '[%(asctime)s] [%(request_id)s] [%(filename)s:%(lineno)d] [%(ip_addr)s] SERVER_LOG_%(levelname)s: %(message)s'
)

logger = logging.getLogger('my_custom_logger')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setLevel(logging.INFO)
file_handler.addFilter(RequestIDLogFilter())
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.addFilter(RequestIDLogFilter())
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
