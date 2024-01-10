from typing import Dict

from modelscope_agent import prompt_generator_register
from modelscope_agent.agent_types import AgentType
from modelscope_agent.constant import DEFAULT_MODEL_CONFIG
from modelscope_agent.llm import LLM
from modelscope_agent.utils.logger import agent_logger as logger

from .messages_prompt import MessagesGenerator
from .mrkl_prompt import MrklPromptGenerator
from .ms_prompt import MSPromptGenerator


class PromptGeneratorFactory:

    @classmethod
    def get_prompt_generator(cls,
                             agent_type: AgentType = AgentType.DEFAULT,
                             model: LLM = None,
                             **kwargs):
        uuid = kwargs.get('uuid', 'default_user')
        logger.info(
            uuid=uuid,
            message='Initiating prompt generator.',
            content={
                'agent_type': agent_type,
                'model': str(model),
                'kwargs': str(kwargs)
            })

        prompt_generator = kwargs.get('prompt_generator', None)
        if prompt_generator:
            return cls._string_to_obj(prompt_generator, llm=model, **kwargs)

        if model:
            language = kwargs.pop('language', 'en')
            prompt_generator = cls._get_model_default_type(
                model, language, uuid)
            if prompt_generator:
                return prompt_generator(llm=model, **kwargs)

        return cls._get_prompt_generator_by_agent_type(
            agent_type, llm=model, **kwargs)

    def _string_to_obj(prompt_generator_name: str, **kwargs):
        uuid = kwargs.get('uuid', 'default_user')
        for name, generator in prompt_generator_register.registered.items():
            if prompt_generator_name == name:
                obj = generator(**kwargs)
                return obj
        logger.error(
            uuid=uuid,
            message=
            f'prompt generator {prompt_generator_name} is not registered.',
            content={'registered': str(prompt_generator_register.registered)})
        raise ValueError(
            f'prompt generator {prompt_generator_name} is not registered. prompt_generator_register.registered: \
              {prompt_generator_register.registered}')

    def _get_model_default_type(model: LLM,
                                language: str = 'en',
                                uuid: str = 'default_user'):
        if not issubclass(model.__class__, LLM):
            return None
        model_id = model.model_id
        if not model_id:
            logger.warning(uuid=uuid, message=f'llm has no name: {model}')
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
        logger.warning(
            uuid=uuid,
            message='prompt generator cannot initiated by model type.',
            content={
                'model_id': model_id,
                'candidate': str(candidate),
                'model_with_default_type': str(DEFAULT_MODEL_CONFIG.keys())
            })
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


get_prompt_generator = PromptGeneratorFactory.get_prompt_generator
