import os
from enum import Enum
from typing import Dict, Optional, Union

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


class ToolInstance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    status: Optional[str]  # including "pending", "running", "exited", "failed"
    image: Optional[str] = None
    tenant_id: Optional[str] = None
    container_id: Optional[str] = None
    ip: Optional[str] = None
    port: Optional[int] = 31513
    error: Optional[str] = None


class ToolRegisterInfo(BaseModel):
    node_name: str
    image: str = ''
    workspace_dir: str = os.getcwd()
    tool_name: str
    tenant_id: str
    config: Dict = {}
    port: Optional[int] = 31513
    tool_url: str = ''


class CreateTool(BaseModel):
    tool_name: str
    tenant_id: str = 'default'
    tool_cfg: Dict = {}
    tool_image: str = 'modelscope-agent/tool-node:latest'
    tool_url: str = ''


class ExecuteTool(BaseModel):
    tool_name: str
    tenant_id: str = 'default'
    params: str = ''
    kwargs: Dict = {}


class ExecuteOpenAPISchema(BaseModel):
    openapi_name: str = ''
    url: str = ''
    params: Union[str, Dict] = ''
    headers: Dict = {}
    method: str = 'GET'
    data: Dict = {}
    cookies: Dict = {}


class ContainerStatus(Enum):
    pending = 'pending'
    running = 'running'
    exited = 'exited'
    failed = 'failed'
