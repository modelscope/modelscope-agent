from modelscope_agent.agent_types import AgentType

from .messages_prompt import MessagesGenerator
from .mrkl_prompt import MrklPromptGenerator
from .ms_prompt import MSPromptGenerator


def get_prompt_generator(agent_type: AgentType = AgentType.DEFAULT, **kwargs):
    if AgentType.DEFAULT == agent_type or agent_type == AgentType.MS_AGENT:
        return MSPromptGenerator(**kwargs)
    elif AgentType.MRKL == agent_type:
        return MrklPromptGenerator(**kwargs)
    elif AgentType.Messages == agent_type:
        return MessagesGenerator(**kwargs)
    else:
        raise NotImplementedError
