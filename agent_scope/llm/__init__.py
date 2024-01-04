import re

from .base import LLM_REGISTRY, BaseChatModel
from .openai import QwenChatAsOAI
from .qwen_dashscope import QwenChatAtDS


def get_chat_model(model: str, model_server: str, **kwargs) -> BaseChatModel:
    """
    model: the model name: such as qwen-max, gpt-4 ...
    model_server: the source of model, such as dashscope, openai, modelscope ...
    **kwargs: more parameters, such as api_key, api_base
    """
    model_type = re.split(r'[-/_]', model)[0]  # parser qwen / gpt / ...
    registered_model_id = f'{model_type}_{model_server}'
    if registered_model_id in LLM_REGISTRY:  # specific model from specific source
        return LLM_REGISTRY[registered_model_id](model, model_server, **kwargs)
    elif model_server in LLM_REGISTRY:  # specific source
        return LLM_REGISTRY[model_server](model, model_server, **kwargs)
    else:
        raise NotImplementedError
