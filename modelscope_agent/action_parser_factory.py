from typing import Dict

from modelscope_agent import action_parser_register
from modelscope_agent.agent_types import AgentType
from modelscope_agent.constant import DEFAULT_MODEL_CONFIG
from modelscope_agent.llm import LLM

from .action_parser import (MRKLActionParser, MsActionParser,
                            OpenAiFunctionsActionParser)


class ActionParserFactory:

    @classmethod
    def get_action_parser(cls,
                          agent_type: AgentType = AgentType.DEFAULT,
                          model: LLM = None,
                          cfg: Dict = None,
                          **kwargs):

        # cfg eg. {"prompt_generator": "MessagesGenerator", "action_parser": "MRKLActionParser"}
        if cfg:
            action_parser = cfg.get('action_parser', None)
            print('action_parser: {action_parser}')
            if action_parser:
                return cls._string_to_obj(cls, action_parser, **kwargs)

        print(
            f'agent_type: {agent_type}, model: {model}, cfg: {cfg}, **kwargs : {kwargs}'
        )

        if model:
            language = kwargs.pop('language', 'en')
            action_parser = cls._get_model_default_type(cls, model, language)
            if action_parser:
                return cls._string_to_obj(cls, action_parser, **kwargs)
        print(
            f'2agent_type: {agent_type}, model: {model}, cfg: {cfg}, **kwargs : {kwargs}'
        )

        return cls._get_action_parser_by_agent_type(cls, agent_type, **kwargs)

    def _string_to_obj(cls, action_parser_name: str, **kwargs):
        print(
            f'action_parser_register.registered: {action_parser_register.registered}'
        )
        for parser in action_parser_register.registered:
            print(
                f'action_parser_name: {action_parser_name}, parser.__name__: {parser.__name__}'
            )
            if action_parser_name == parser.__name__:
                obj = parser(**kwargs)
                return obj
        raise ValueError(
            f'prompt parser {action_parser_name} is not registered.')

    def _get_model_default_type(cls, model: LLM, language: str = 'en'):
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
                return model_cfg[language].get('action_parser', None)
            return model_cfg.get('action_parser', None)
        return None

    def _get_action_parser_by_agent_type(
            cls, agent_type: AgentType = AgentType.DEFAULT):
        if AgentType.DEFAULT == agent_type or agent_type == AgentType.MS_AGENT:
            return MsActionParser()
        elif AgentType.MRKL == agent_type:
            return MRKLActionParser()
        elif AgentType.Messages == agent_type:
            return OpenAiFunctionsActionParser()
        else:
            raise NotImplementedError


get_action_parser = ActionParserFactory.get_action_parser