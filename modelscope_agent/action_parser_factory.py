from typing import Dict

from modelscope_agent import action_parser_register
from modelscope_agent.agent_types import AgentType
from modelscope_agent.constant import DEFAULT_MODEL_CONFIG
from modelscope_agent.llm import LLM
from modelscope_agent.utils.logger import agent_logger as logger

from .action_parser import (MRKLActionParser, MsActionParser,
                            OpenAiFunctionsActionParser)


class ActionParserFactory:

    @classmethod
    def get_action_parser(cls,
                          agent_type: AgentType = AgentType.DEFAULT,
                          model: LLM = None,
                          **kwargs):
        logger.info(
            uuid=uuid,
            message='Initiating action parser.',
            content={
                'agent_type': agent_type,
                'model': model,
                'kwargs': kwargs
            })
        action_parser = kwargs.get('action_parser', None)
        if action_parser:
            return cls._string_to_obj(action_parser, **kwargs)

        if model:
            language = kwargs.pop('language', 'en')
            action_parser = cls._get_model_default_type(model, language)
            if action_parser:
                return cls._string_to_obj(action_parser, **kwargs)

        return cls._get_action_parser_by_agent_type(agent_type, **kwargs)

    def _string_to_obj(action_parser_name: str, **kwargs):
        for name, parser in action_parser_register.registered.items():
            if action_parser_name == name:
                obj = parser(**kwargs)
                return obj
        raise ValueError(
            uuid=uuid,
            message=f'action parser {action_parser_name} is not registered.',
            content={'registered': action_parser_register.registered})

    def _get_model_default_type(model: LLM, language: str = 'en'):
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
                return model_cfg[language].get('action_parser', None)
            return model_cfg.get('action_parser', None)
        logger.warning(
            uuid=uuid,
            message='action parser cannot initiated by model type.',
            content={
                'model_id': model_id,
                'candidate': candidate,
                'model_with_default_config': DEFAULT_MODEL_CONFIG.keys()
            })
        return None

    def _get_action_parser_by_agent_type(
            agent_type: AgentType = AgentType.DEFAULT, **kwargs):
        if AgentType.DEFAULT == agent_type or agent_type == AgentType.MS_AGENT:
            return MsActionParser(**kwargs)
        elif AgentType.MRKL == agent_type:
            return MRKLActionParser(**kwargs)
        elif AgentType.Messages == agent_type:
            return OpenAiFunctionsActionParser(**kwargs)
        else:
            raise NotImplementedError


get_action_parser = ActionParserFactory.get_action_parser
