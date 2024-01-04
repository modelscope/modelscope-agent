import re

from .base import LLM_REGISTRY, BaseChatModel
from .custom import CustomLLM
from .dashscope import DashScopeLLM
from .dashscope_qwen import QwenChatAtDS
from .modelscope import ModelScopeLLM
from .modelscope_chatglm import ModelScopeChatGLM
from .openai import OpenAi


def get_chat_model(model: str, model_server: str, **kwargs) -> BaseChatModel:
    """
    model: the model name: such as qwen-max, gpt-4 ...
    model_server: the source of model, such as dashscope, openai, modelscope ...
    **kwargs: more parameters, such as api_key, api_base
    """
    model_type = re.split(r'[-/_]', model)[0]  # parser qwen / gpt / ...
    registered_model_id = f'{model_server}_{model_type}'
    if registered_model_id in LLM_REGISTRY:  # specific model from specific source
        return LLM_REGISTRY[registered_model_id](model, model_server, **kwargs)
    elif model_server in LLM_REGISTRY:  # specific source
        return LLM_REGISTRY[model_server](model, model_server, **kwargs)
    else:
        raise NotImplementedError


__all__ = [
    'LLM_REGISTRY', 'BaseChatModel', 'OpenAi', 'DashScopeLLM', 'QwenChatAtDS',
    'ModelScopeLLM', 'ModelScopeChatGLM', 'CustomLLM'
]
