def get_llm_cls(llm_type):
    if llm_type == 'dashscope':
        from .dashscope_llm import DashScopeLLM
        return DashScopeLLM
    elif llm_type == 'openai':
        from .openai import OpenAi
        return OpenAi
    elif llm_type == 'modelscope':
        from .modelscope_llm import ModelScopeLLM
        return ModelScopeLLM
    else:
        raise ValueError(f'Invalid llm_type {llm_type}')


class LLMFactory:

    @staticmethod
    def build_llm(llm_type, cfg):
        llm_cls = get_llm_cls(llm_type)
        llm_cfg = cfg.get(llm_type, {})
        return llm_cls(cfg=llm_cfg)
