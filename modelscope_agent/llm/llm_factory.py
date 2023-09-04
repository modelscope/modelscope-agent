def get_llm_cls(llm_type):
    if llm_type == 'ms_gpt':
        from .ms_gpt import ModelScopeGPT
        return ModelScopeGPT
    elif llm_type == 'openai':
        from .openai import OpenAi
        return OpenAi
    else:
        from .local_llm import LocalLLM
        return LocalLLM


class LLMFactory:

    @staticmethod
    def build_llm(llm_type, cfg):
        llm_cls = get_llm_cls(llm_type)
        llm_cfg = cfg.get(llm_type, {})
        return llm_cls(cfg=llm_cfg)
