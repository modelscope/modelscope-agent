from typing import Dict, List, Optional, Union

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class Tool(BaseModel):
    name: str = Field(..., title='Tool name')
    description: str = Field(..., title='Tool description')
    parameters: List[Dict] = Field([], title='List of parameters')


class LLMConfig(BaseModel):
    model: str = Field(..., title='Model name')
    model_server: str = Field(..., title='Model source')
    api_key: str = Field(None, title='API key')
    generate_config: dict = Field({}, title='Model config')


class AgentConfig(BaseModel):
    name: str = Field(..., title='Agent name')
    description: str = Field(..., title='Agent description')
    instruction: str = Field(..., title='Agent instruction')
    tools: List[Union[str, Tool]] = Field([], title='List of tools')


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., title='List of messages')
    llm_config: LLMConfig = Field(..., title='LLM config')
    agent_config: AgentConfig = Field(None, title='Agent config')
    stream: bool = Field(False, title='Stream output')
    use_knowledge: bool = Field(False, title='Whether to use knowledge')
    files: List[str] = Field([], title='List of files used in knowledge')
    uuid_str: Optional[str] = Field('test', title='UUID string')
    tools: List[Dict] = Field(None, title='Tools config')
    tool_choice: Optional[str] = Field('auto', title='tool usage choice')
    use_tool_api: Optional[bool] = Field(False, title='use tool api or not')
    kwargs: Dict[str, object] = Field({},
                                      title='store additional key to kwargs')

    def __init__(__pydantic_self__, **data):
        # store all additional keys to `kwargs`
        extra_keys = set(data.keys()) - set(
            __pydantic_self__.model_fields.keys())
        kwargs = {key: data.pop(key) for key in extra_keys}
        super().__init__(**data)
        __pydantic_self__.kwargs = kwargs

    class Config:
        extra = 'allow'


class ToolResponse(BaseModel):
    name: str = Field(..., title='Tool name')
    inputs: Dict = Field({}, title='List of inputs')


class ChatResponse(BaseModel):
    response: str = Field(..., title='Response message')
    require_actions: bool = Field(False, title='Whether require actions')
    tool: ToolResponse = Field(None, title='Tool response')
