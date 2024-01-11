from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    """
    Message: message information
    """
    role: str = ''
    content: str = ''


class Document(BaseModel):
    """
    Document: Record User uploaded document information
    """
    url: str
    time: str
    source: str
    raw: list
    title: str
    topic: str
    checked: bool
    session: list


class AgentHolder(BaseModel):
    """
    AgentHolder: Record Agent information
    """
    session: str = ''
    uuid: str = ''
    history: List[Message] = []
    knowledge: List = ''  # in case retrieval cost is much higher than storage
