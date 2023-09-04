from .local_llm import LocalLLM
from .ms_gpt import ModelScopeGPT
from .openai import OpenAi

LLM_MAPPING = {'ms_gpt': ModelScopeGPT, 'openai': OpenAi}


class LLMFactory:

    @staticmethod
    def build_llm(name, cfg):
        llm_cls = LLM_MAPPING.get(name, LocalLLM)
        llm_cfg = cfg.get(name, {})
        return llm_cls(cfg=llm_cfg)
