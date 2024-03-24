from pydantic import BaseModel


class ToolRequest(BaseModel):
    params: str
    messages: list = []


class ToolResponse(BaseModel):
    result: str
    messages: list = []
