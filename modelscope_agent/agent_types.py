from enum import Enum


class AgentType(str, Enum):

    DEFAULT = 'default'
    """"""

    MS_AGENT = 'ms-agent'
    """An agent that uses the ModelScope-agent specific format does a reasoning step before acting .
    """

    MRKL = 'mrkl'
    """An agent that does a reasoning step before acting with mrkl"""

    REACT = 'react'
    """An agent that does a reasoning step before acting with react"""

    OPENAI_FUNCTIONS = 'openai-functions'
    """An agent optimized for using open AI functions."""
