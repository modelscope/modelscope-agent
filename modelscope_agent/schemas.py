from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    role: str = ''
    content: str = ''


class AgentHolder(BaseModel):
    session: str = ''
    uuid: str = ''
    history: List[Message] = []
    knowledge: List = ''  # in case retrieval cost is much higher than storage
