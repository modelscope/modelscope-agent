import logging
from contextvars import ContextVar

request_id_var = ContextVar("request_id", default="")


# 创建一个日志过滤器，用于加入request_id到日志记录中
class RequestIDLogFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get("")
        return True


# 设置日志格式
formatter = logging.Formatter(
    '[%(asctime)s] [%(request_id)s] [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'
)

# 创建和配置日志记录器
logger = logging.getLogger('my_custom_logger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.addFilter(RequestIDLogFilter())
logger.addHandler(handler)
