import os
from contextlib import asynccontextmanager

from connections import (ToolInstance, ToolRegistration, create_db_and_tables,
                         engine, get_session)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from sandbox import (NODE_NETWORK, NODE_PORT, restart_docker_container,
                     start_docker_container)
from sqlmodel import Session, select


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load tool from modelscope-agent
    """
    Startup function to initialize the required services and variables for the application.
    """
    try:
        # start docker service
        os.system('tool manager service start')
    except Exception:
        pass
    app.containers_info = {}
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


def start_docker_container_and_store_status(tool: ToolRegistration):
    with Session(engine) as session:
        # 先创建一个记录表示此任务处于 pending 状态
        tool_container = ToolInstance(name=tool.name, status='pending')
        session.add(tool_container)
        session.commit()
        app.containers_info[tool.name] = {'status': 'pending'}

        try:
            container = start_docker_container(tool)
            ip = container.attrs['NetworkSettings'][NODE_NETWORK]['IPAddress']
            port = NODE_PORT
            tool_container.container_id = container.id
            tool_container.status = container.status
            tool_container.ip = ip
            tool_container.port = port
            app.containers_info[tool.name] = {
                'status': container.status,
                'container_id': container.id
            }
            session.add(tool_container)
            session.commit()
        except Exception as e:
            # if docker start failed, record the error
            tool_container.status = 'failed'
            tool_container.error = str(e)
            app.containers_info[tool.name] = {
                'status': 'failed',
                'error': str(e)
            }
            session.add(tool_container)
            session.commit()


def restart_docker_container_and_update_status(tool: ToolRegistration):
    with Session(engine) as session:
        tool_container = session.exec(
            select(ToolInstance).where(
                ToolInstance.name == tool.name)).first()
        if not tool_container:
            raise HTTPException(status_code=404, detail='Tool not found')
        try:
            container = restart_docker_container(tool)
            ip = container.attrs['NetworkSettings'][NODE_NETWORK]['IPAddress']
            port = NODE_PORT
            tool_container.container_id = container.id
            tool_container.status = container.status
            tool_container.ip = ip
            tool_container.port = port
            app.containers_info[tool.name] = {
                'status': container.status,
                'container_id': container.id
            }
            session.add(tool_container)
            session.commit()
        except Exception as e:
            # if docker start failed, record the error
            tool_container.status = 'failed'
            tool_container.error = str(e)
            app.containers_info[tool.name] = {
                'status': 'failed',
                'error': str(e)
            }
            session.add(tool_container)
            session.commit()


@app.post('/create_tool_service/')
async def create_tool_service(
    tool_name: str,
    background_tasks: BackgroundTasks,
    tenant_id: str = 'default',
    tool_image: str = 'xxx',
):
    tool_node_name = f'{tool_name}_{tenant_id}'
    tool_register = ToolRegistration(
        name=tool_node_name,
        tenant_id=tenant_id,
        image=tool_image,
        workspace_dir=os.getcwd(),
    )
    background_tasks.add_task(start_docker_container_and_store_status,
                              tool_register)

    return {'tool_node_name': tool_node_name, 'status': 'pending'}


@app.post('/update_tool_service/')
async def update_tool_service(
    tool_name: str,
    background_tasks: BackgroundTasks,
    tenant_id: str = 'default',
    tool_image: str = 'xxx',
):
    tool_node_name = f'{tool_name}_{tenant_id}'
    tool_register = ToolRegistration(
        name=tool_node_name,
        tenant_id=tenant_id,
        image=tool_image,
        workspace_dir=os.getcwd(),
    )
    background_tasks.add_task(restart_docker_container_and_update_status,
                              tool_register)

    return {'tool_node_name': tool_node_name, 'status': 'pending'}


@app.post('/remove_tool/')
async def deregister_tool(tool_id: str,
                          session: Session = Depends(get_session)):

    return {'message': 'Tool deregistered successfully'}


@app.get('/tools/')
async def list_tools(tenant_id: str = 'default'):
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.tenant_id == tenant_id)
        tools = session.exec(statement)
    return tools


@app.post('/get_tool_service_url/')
async def get_tool_service_url(tool_name: str, tenant_id: str = 'default'):

    # get tool instance
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.name == f'{tool_name}_{tenant_id}')
        results = session.exec(statement)
        tool_instance = results.first()
    if not tool_instance:
        raise HTTPException(status_code=404, detail='Tool not found')

    # get tool service url
    try:
        tool_service_url = 'http://' + tool_instance.ip + ':' + str(
            tool_instance.port) + '/execute_tool'
        return tool_service_url
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=
            f'Failed to get tool service url, with error {tool_instance.error}'
        )
