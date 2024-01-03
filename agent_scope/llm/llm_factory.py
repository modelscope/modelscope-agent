def get_llm_cls(llm_type, model_name):
    if llm_type == 'dashscope':
        from .dashscope_llm import DashScopeLLM
        return DashScopeLLM
    elif llm_type == 'custom_llm':
        from .custom_llm import CustomLLM
        return CustomLLM
    elif llm_type == 'openai':
        from .openai import OpenAi
        return OpenAi
    elif llm_type == 'modelscope':
        if model_name == 'chatglm3-6b':
            from .modelscope_llm import ModelScopeChatGLM
            return ModelScopeChatGLM
        from .modelscope_llm import ModelScopeLLM
        return ModelScopeLLM
    else:
        raise ValueError(f'Invalid llm_type {llm_type}')


class LLMFactory:

    @staticmethod
    def build_llm(model_name, cfg):
        llm_type = cfg[model_name].pop('type')
        llm_cls = get_llm_cls(llm_type, model_name)
        llm_cfg = cfg[model_name]
        return llm_cls(cfg=llm_cfg)
