from pydantic import BaseModel


class ToolRequest(BaseModel):
    params: str
    kwargs: dict = {}
    messages: list = []


class ToolResponse(BaseModel):
    result: str
    messages: list = []
