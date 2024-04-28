from typing import Dict, List, Optional

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = Field(..., title='Model name')
    model_server: str = Field(..., title='Model source')
    api_key: str = Field(None, title='API key')
    generate_config: dict = Field({}, title='Model config')


class AgentConfig(BaseModel):
    name: str = Field(..., title='Agent name')
    description: str = Field(..., title='Agent description')
    instruction: str = Field(..., title='Agent instruction')
    tools: List[str] = Field([], title='List of tools')


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., title='List of messages')
    llm_config: LLMConfig = Field(..., title='LLM config')
    agent_config: AgentConfig = Field(..., title='Agent config')
    stream: bool = Field(False, title='Stream output')
    use_knowledge: bool = Field(False, title='Whether to use knowledge')
    files: List[str] = Field([], title='List of files used in knowledge')
    uuid_str: Optional[str] = Field('test', title='UUID string')


# for upper api
class AgentModel(BaseModel):
    uuid: str = Field(..., title='Agent ID')
    llm_config: LLMConfig = Field(..., title='LLM config')
    agent_config: AgentConfig = Field(..., title='Agent config')


class MemoryModel(BaseModel):
    uuid: str = Field(..., title='Memory ID')
    history: List[Dict[str, str]] = Field([], title='List of messages')
    files: List[str] = Field([], title='List of files used in knowledge')
