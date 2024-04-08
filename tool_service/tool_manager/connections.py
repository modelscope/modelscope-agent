import os
from enum import Enum
from typing import Optional

import docker
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine

# database setting
DATABASE_DIR = os.getenv('DATABASE_DIR', './test.db')
DATABASE_URL = os.path.join('sqlite:///', DATABASE_DIR)

engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def drop_db_and_tables():
    SQLModel.metadata.drop_all(engine)


def get_docker_client():
    # Initialize docker client. Throws an exception if Docker is not reachable.
    try:
        docker_client = docker.from_env()
    except docker.errors.DockerException as e:
        print('Please check Docker is running using `docker ps`.')
        print(f'Error! {e}', flush=True)
        raise e
    return docker_client


def get_session():
    with Session(engine) as session:
        yield session


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
    config: dict = {}
    port: Optional[int] = 31513


class CreateTool(BaseModel):
    tool_name: str
    tenant_id: str = 'default'
    tool_cfg: dict = {}
    tool_image: str = 'modelscope-agent/tool-node:v0.4'


class GetToolUrl(BaseModel):
    tool_name: str
    tenant_id: str = 'default'


class ContainerStatus(Enum):
    pending = 'pending'
    running = 'running'
    exited = 'exited'
    failed = 'failed'
