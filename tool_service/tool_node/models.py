from pydantic import BaseModel


class ToolRequest(BaseModel):
    params: str
    kwargs: dict = {}
    messages: list = []
    request_id: str


class ToolResponse(BaseModel):
    result: str
    messages: list = []
