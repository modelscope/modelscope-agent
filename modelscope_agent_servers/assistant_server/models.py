import time
from typing import Dict, List, Literal, Optional, Union

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., title='Role name')
    content: Union[str, List[Dict[str, str]]] = Field(...,
                                                      title='Message content')
    tool_calls: Optional[List[Dict]] = Field(None, title='Tool calls')


class DeltaMessage(BaseModel):
    role: str = Field(None, title='Role name')
    content: str = Field(None, title='Message content')


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


class AgentRequest(BaseModel):
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


class AgentResponse(BaseModel):
    response: str = Field(..., title='Response message')
    require_actions: bool = Field(False, title='Whether require actions')
    tool: ToolResponse = Field(None, title='Tool response')


class Usage(BaseModel):
    prompt_tokens: int = Field(-1, title='Prompt tokens consumed')
    completion_tokens: int = Field(-1, title='Completion tokens consumed')
    total_tokens: int = Field(-1, title='Total tokens consumed')


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., title='Model name')
    messages: List[ChatMessage]
    tools: Optional[List[Dict]] = Field(None, title='Tools config')
    tool_choice: Optional[str] = Field('auto', title='tool usage choice')
    parallel_tool_calls: Optional[bool] = Field(
        True,
        title='Whether to enable parallel function calling during tool use.')
    stream: Optional[bool] = Field(False, title='Stream output')
    user: str = Field('default_user', title='User name')


class ChatCompletionResponseChoice(BaseModel):
    index: int = Field(..., title='Index of the choice')
    message: ChatMessage = Field(..., title='Chat message')
    finish_reason: str = Field(..., title='Finish reason')


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = Field(..., title='Index of the choice')
    delta: DeltaMessage = Field(..., title='Chat message')
    finish_reason: str = Field(None, title='Finish reason')


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., title='Unique id for chat completion')
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., title='Model name')
    system_fingerprint: str = Field(None, title='Cuurently request id')
    object: Literal['chat.completion', 'chat.completion.chunk'] = Field(
        'chat.completion', title='Object type')
    usage: Optional[Usage] = Field(
        default=Usage(), title='Token usage information')
