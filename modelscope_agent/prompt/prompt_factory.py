from typing import Dict

from modelscope_agent import prompt_generator_register
from modelscope_agent.agent_types import AgentType
from modelscope_agent.constant import DEFAULT_MODEL_CONFIG
from modelscope_agent.llm import LLM

from .messages_prompt import MessagesGenerator
from .mrkl_prompt import MrklPromptGenerator
from .ms_prompt import MSPromptGenerator


class PromptGeneratorFactory:

    @classmethod
    def get_prompt_generator(cls,
                             agent_type: AgentType = AgentType.DEFAULT,
                             model: LLM = None,
                             **kwargs):
        print(f'agent_type: {agent_type}, model: {model}, **kwargs : {kwargs}')

        prompt_generator = kwargs.get('prompt_generator', None)
        if prompt_generator:
            print('prompt_generator: {prompt_generator}')
            return cls._string_to_obj(prompt_generator, llm=model, **kwargs)

        if model:
            language = kwargs.pop('language', 'en')
            prompt_generator = cls._get_model_default_type(model, language)
            print('prompt_generator: {prompt_generator}')
            if prompt_generator:
                return cls._string_to_obj(
                    prompt_generator, llm=model, **kwargs)

        return cls._get_prompt_generator_by_agent_type(
            agent_type, llm=model, **kwargs)

    def _string_to_obj(prompt_generator_name: str, **kwargs):
        print(
            f'prompt_generator_register.registered: {prompt_generator_register.registered}'
        )
        for generator in prompt_generator_register.registered:
            print(
                f'prompt_generator_name: {prompt_generator_name}, generator.__name__: {generator.__name__}'
            )
            if prompt_generator_name == generator.__name__:
                obj = generator(**kwargs)
                return obj
        raise ValueError(
            f'prompt generator {prompt_generator_name} is not registered.')

    def _get_model_default_type(model: LLM, language: str = 'en'):
        if not issubclass(model.__class__, LLM):
            return None
        model_id = model.model_id
        print(f'model: {model_id}')
        if not model_id:
            # logger.warning
            print(f'model has no name: {model}')
            return None

        candidate = []
        for key in DEFAULT_MODEL_CONFIG.keys():
            if model_id.startswith(key):
                candidate.append(key)

        full_model_id = max(candidate, key=len, default=None)
        if full_model_id:
            model_cfg = DEFAULT_MODEL_CONFIG.get(full_model_id, {})
            if model_cfg.get(language, None):
                return model_cfg[language].get('prompt_generator', None)
            return model_cfg.get('prompt_generator', None)
        return None

    def _get_prompt_generator_by_agent_type(
            agent_type: AgentType = AgentType.DEFAULT, **kwargs):
        if AgentType.DEFAULT == agent_type or agent_type == AgentType.MS_AGENT:
            return MSPromptGenerator(**kwargs)
        elif AgentType.MRKL == agent_type:
            return MrklPromptGenerator(**kwargs)
        elif AgentType.Messages == agent_type:
            return MessagesGenerator(**kwargs)
        else:
            raise NotImplementedError
