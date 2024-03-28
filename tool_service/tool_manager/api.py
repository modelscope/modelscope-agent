import uuid
from typing import Optional

from docker import DockerClient
from fastapi import Depends, FastAPI, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine

API_KEY = 'your_api_key'
API_KEY_NAME = 'access_token'

app = FastAPI()
docker_client = DockerClient(base_url='unix://var/run/docker.sock')

# 数据库配置
DATABASE_URL = 'sqlite:///./test.db'
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})


# 数据库模型
class ToolInstance(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    name: str
    image: str
    container_id: str


# 创建数据库表
SQLModel.metadata.create_all(engine)


# 获取数据库 session 的依赖
def get_session():
    with Session(engine) as session:
        yield session


# 同步 Docker 容器与数据库的状态
def sync_docker_containers_with_db():
    with Session(engine) as session:
        containers = docker_client.containers.list(all=True)
        for container in containers:
            tool_instance = session.query(ToolInstance).filter(
                ToolInstance.container_id == container.id).first()
            if not tool_instance:
                # 如果容器在数据库中没有记录，停止并移除该容器
                container.remove(force=True)


@app.on_event('startup')
async def startup_event():
    sync_docker_containers_with_db()


@app.post('/register_tool/')
async def register_tool(tool_name: str,
                        tool_image: str,
                        session: Session = Depends(get_session)):
    # 从 Docker 镜像库拉取镜像并创建新的容器实例
    container = docker_client.containers.run(tool_image, detach=True)
    tool_id = str(uuid.uuid4())
    tool_instance = ToolInstance(
        id=tool_id,
        name=tool_name,
        image=tool_image,
        container_id=container.id)
    session.add(tool_instance)
    session.commit()
    return {'tool_id': tool_id}


@app.post('/run_tool/')
async def run_tool(tool_id: str,
                   payload: dict,
                   session: Session = Depends(get_session)):
    # 在指定的工具实例容器中执行命令
    tool_instance = session.query(ToolInstance).filter(
        ToolInstance.id == tool_id).first()
    if not tool_instance:
        raise HTTPException(status_code=404, detail='Tool not found')

    container = docker_client.containers.get(tool_instance.container_id)
    exec_result = container.exec_run('echo Hello World')
    return {'result': exec_result.output.decode('utf-8')}


@app.post('/deregister_tool/')
async def deregister_tool(tool_id: str,
                          session: Session = Depends(get_session)):
    # 注销工具实例并移除对应的 Docker 容器
    tool_instance = session.query(ToolInstance).filter(
        ToolInstance.id == tool_id).first()
    if not tool_instance:
        raise HTTPException(status_code=404, detail='Tool not found')

    container = docker_client.containers.get(tool_instance.container_id)
    container.remove(force=True)
    session.delete(tool_instance)
    session.commit()
    return {'message': 'Tool deregistered successfully'}


@app.get('/tools/')
async def list_tools(session: Session = Depends(get_session)):
    # 列出所有注册的工具实例
    tools = session.query(ToolInstance).all()
    return tools


@app.post('/get_tool/')
async def get_tool(tool_id: str, session: Session = Depends(get_session)):
    # 根据 ID 获取特定工具实例的详细信息
    tool_instance = session.query(ToolInstance).filter(
        ToolInstance.id == tool_id).first()
    if not tool_instance:
        raise HTTPException(status_code=404, detail='Tool not found')
    return tool_instance
