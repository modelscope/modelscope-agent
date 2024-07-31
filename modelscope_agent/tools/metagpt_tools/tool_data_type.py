# this code is originally from https://github.com/geekan/MetaGPT
from pydantic import BaseModel


class ToolSchema(BaseModel):
    description: str


class Tool(BaseModel):
    name: str
    path: str
    schemas: dict = {}
    code: str = ''
    tags: list[str] = []
