import os
from typing import Optional

import docker
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine

API_KEY = 'your_api_key'
API_KEY_NAME = 'access_token'

# 数据库配置
DATABASE_URL = 'sqlite:///./test.db'
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


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
    id: Optional[str] = Field(default=None, primary_key=True)
    name: str
    image: str
    tenant_id: str
    status: str  # including "pending", "running", "exited", "failed"
    container_id: str
    ip: str
    port: int = 31513


class ToolRegistration(BaseModel):
    name: str
    image: str = ''
    workspace_dir: str = os.getcwd()
