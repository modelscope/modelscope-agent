from modelscope_agent.agent_types import AgentType

from .mrkl_prompt import MrklPromptGenerator
from .ms_prompt import MSPromptGenerator
from .openai_function_prompt import OpenAiFunctionsPromptGenerator


def get_prompt_generator(agent_type: AgentType = AgentType.DEFAULT, **kwargs):
    if AgentType.DEFAULT == agent_type or agent_type == AgentType.MS_AGENT:
        return MSPromptGenerator(**kwargs)
    elif AgentType.MRKL == agent_type:
        return MrklPromptGenerator(**kwargs)
    elif AgentType.OPENAI_FUNCTIONS == agent_type:
        return OpenAiFunctionsPromptGenerator(**kwargs)
    else:
        raise NotImplementedError
