def get_llm_cls(llm_type):
    if llm_type == 'dashscope_llm':
        from .dashscope_llm import DashScopeLLM
        return DashScopeLLM
    elif llm_type == 'openai':
        from .openai import OpenAi
        return OpenAi
    else:
        from .modelscope_llm import ModelScopeLLM
        return ModelScopeLLM


class LLMFactory:

    @staticmethod
    def build_llm(llm_type, cfg):
        llm_cls = get_llm_cls(llm_type)
        llm_cfg = cfg.get(llm_type, {})
        return llm_cls(cfg=llm_cfg)
