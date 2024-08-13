import re

from .base import LLM_REGISTRY, BaseChatModel
from .dashscope import DashScopeLLM, QwenChatAtDS
from .modelscope import ModelScopeChatGLM, ModelScopeLLM
from .ollama import OllamaLLM
from .openai import OpenAi
from .vllm import VllmLLM
from .zhipu import ZhipuLLM


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
    'ModelScopeLLM', 'ModelScopeChatGLM', 'ZhipuLLM', 'OllamaLLM', 'VllmLLM'
]
