from .base import LLM_REGISTRY, BaseChatModel
from .qwen_dashscope import QwenChatAtDS
from .qwen_oai import QwenChatAsOAI


def get_chat_model(model: str, model_server: str, **kwargs) -> BaseChatModel:
    """
    model: the model name: such as qwen-max, gpt-4 ...
    model_server: the source of model, such as dashscope, oai, modelscope ...
    **kwargs: more parameters, such as api_key
    """
    model_id = f'{model}_{model_server}'
    if model_id in LLM_REGISTRY:  # specific model from specific source
        return LLM_REGISTRY[model_id](model, model_server, **kwargs)
    elif model_server in LLM_REGISTRY:  # specific source
        return LLM_REGISTRY[model_id](model, model_server, **kwargs)
    else:
        raise NotImplementedError
