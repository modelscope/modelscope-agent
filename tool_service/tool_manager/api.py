import os
from contextlib import asynccontextmanager

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from sqlmodel import Session, select
from tool_service.tool_manager.connections import create_db_and_tables, engine
from tool_service.tool_manager.models import (ContainerStatus, CreateTool,
                                              ExecuteTool, GetToolUrl,
                                              ToolInstance, ToolRegisterInfo)
from tool_service.tool_manager.sandbox import (NODE_NETWORK,
                                               remove_docker_container,
                                               restart_docker_container,
                                               start_docker_container)
from tool_service.tool_manager.utils import PortGenerator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load sqlmodel database at startup stage
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
    app.node_port_generator = PortGenerator()
    yield


app = FastAPI(lifespan=lifespan)


def start_docker_container_and_store_status(tool: ToolRegisterInfo,
                                            app_instance: FastAPI):
    with Session(engine) as session:
        # create container for pending record
        tool_container = ToolInstance(
            name=tool.node_name, status=ContainerStatus.pending.value)

        statement = select(ToolInstance).where(
            ToolInstance.name == tool.node_name)
        result = session.exec(statement).first()
        if result is not None:
            # make sure the container restarted if the container was existed or failed
            if result.status in [
                    ContainerStatus.exited.value, ContainerStatus.failed.value
            ]:
                try:
                    remove_docker_container(tool)
                except Exception:
                    pass
                result.status = ContainerStatus.pending.value
                session.add(result)
                session.commit()
                session.refresh(result)
                tool.port = result.port
            else:
                # make sure the container keep running or pending status
                return
        else:
            # record the pending status
            session.add(tool_container)
            session.commit()
            tool_container.port = next(app_instance.node_port_generator)
            tool.port = tool_container.port

        app_instance.containers_info[tool.node_name] = {
            'status': ContainerStatus.pending.value
        }

        try:
            container = start_docker_container(tool)
            statement = select(ToolInstance).where(
                ToolInstance.name == tool_container.name)
            result = session.exec(statement).first()
            tool_container = result

            tool_container.tenant_id = tool.tenant_id
            tool_container.container_id = container.id
            tool_container.status = container.status
            tool_container.error = None
            if NODE_NETWORK == 'host':
                tool_container.ip = 'localhost'
            else:
                tool_container.ip = container.attrs['NetworkSettings'][
                    'Networks'][NODE_NETWORK]['IPAddress']
            app_instance.containers_info[tool.node_name] = {
                'status': container.status,
                'container_id': container.id
            }
            session.add(tool_container)
            session.commit()
        except Exception as e:
            # if docker start failed, record the error
            import traceback
            print(f'error is {traceback.format_exc()}')
            statement = select(ToolInstance).where(
                ToolInstance.name == tool_container.name)
            results = session.exec(statement)
            tool_container = results.one()
            tool_container.status = ContainerStatus.failed.value
            tool_container.error = str(e)
            app_instance.containers_info[tool.node_name] = {
                'status': ContainerStatus.failed.value,
                'error': str(e)
            }
            session.add(tool_container)
            session.commit()


def restart_docker_container_and_update_status(tool: ToolRegisterInfo,
                                               app_instance: FastAPI):
    with Session(engine) as session:
        tool_container = session.exec(
            select(ToolInstance).where(
                ToolInstance.name == tool.node_name)).first()
        if not tool_container:
            raise HTTPException(status_code=404, detail='Tool not found')
        try:
            container = restart_docker_container(tool)
            tool_container.tenant_id = tool.tenant_id
            tool_container.container_id = container.id
            tool_container.status = container.status
            tool_container.ip = container.attrs['NetworkSettings'][
                NODE_NETWORK]['IPAddress']
            tool_container.port = next(app_instance.node_port_generator)
            app_instance.containers_info[tool.node_name] = {
                'status': container.status,
                'container_id': container.id
            }
            session.add(tool_container)
            session.commit()
        except Exception as e:
            # if docker start failed, record the error
            tool_container.status = ContainerStatus.failed.value
            tool_container.error = str(e)
            app_instance.containers_info[tool.node_name] = {
                'status': ContainerStatus.failed.value,
                'error': str(e)
            }
            session.add(tool_container)
            session.commit()


def remove_docker_container_and_update_status(tool: ToolRegisterInfo,
                                              app_instance: FastAPI):
    with Session(engine) as session:
        tool_container = session.exec(
            select(ToolInstance).where(
                ToolInstance.name == tool.node_name)).first()
        if not tool_container:
            raise HTTPException(status_code=404, detail='Tool not found')
        try:
            remove_docker_container(tool)
            tool_container.status = ContainerStatus.exited.value
            app_instance.containers_info[tool.node_name] = {
                'status': ContainerStatus.exited.value,
                'container_id': tool_container.container_id
            }
            # release port
            app_instance.node_port_generator.release(tool_container.port)
            session.add(tool_container)
            session.commit()
        except Exception as e:
            # if docker remove failed, record the error
            tool_container.status = ContainerStatus.exited.value
            app_instance.containers_info[tool.node_name] = {
                'status': ContainerStatus.failed.value,
                'error': str(e)
            }
            app_instance.node_port_generator.release(tool_container.port)
            session.add(tool_container)
            session.commit()


@app.post('/')
@app.get('/')
async def root():
    """
    Root function that returns a message Hello World.

    Returns:
        dict: A dictionary containing a welcoming message.
    """
    return {'message': 'Hello World'}


@app.post('/create_tool_service/')
async def create_tool_service(tool_info: CreateTool,
                              background_tasks: BackgroundTasks):
    # todo: the tool name might be the repo dir for the tool, need to parse in this situation.
    tool_node_name = f'{tool_info.tool_name}_{tool_info.tenant_id}'
    tool_register_info = ToolRegisterInfo(
        node_name=tool_node_name,
        tool_name=tool_info.tool_name,
        config=tool_info.tool_cfg,
        tenant_id=tool_info.tenant_id,
        image=tool_info.tool_image,
        workspace_dir=os.getcwd(),
    )
    background_tasks.add_task(start_docker_container_and_store_status,
                              tool_register_info, app)

    return {
        'tool_node_name': tool_node_name,
        'status': ContainerStatus.pending.value
    }


@app.post('/check_tool_service_status/')
async def check_tool_service_status(
    tool_name: str,
    tenant_id: str = 'default',
):
    # todo: the tool name might be the repo dir for the tool, need to parse in this situation.
    tool_node_name = f'{tool_name}_{tenant_id}'
    with Session(engine) as session:
        tool_container = session.exec(
            select(ToolInstance).where(
                ToolInstance.name == tool_node_name)).first()
        if not tool_container:
            raise HTTPException(status_code=404, detail='Tool not found')
        result = {'status': tool_container.status}

        if tool_container.status == ContainerStatus.failed.value:
            result['error'] = tool_container.error
        return result


@app.post('/update_tool_service/')
async def update_tool_service(
    tool_name: str,
    background_tasks: BackgroundTasks,
    tool_cfg: dict = {},
    tenant_id: str = 'default',
    tool_image: str = 'modelscope-agent/tool-node:latest',
):
    tool_node_name = f'{tool_name}_{tenant_id}'
    tool_register_info = ToolRegisterInfo(
        node_name=tool_node_name,
        tool_name=tool_name,
        config=tool_cfg,
        tenant_id=tenant_id,
        image=tool_image,
        workspace_dir=os.getcwd(),
    )
    background_tasks.add_task(restart_docker_container_and_update_status,
                              tool_register_info, app)

    return {
        'tool_node_name': tool_node_name,
        'status': ContainerStatus.pending.value
    }


@app.post('/remove_tool/')
async def deregister_tool(tool_name: str,
                          background_tasks: BackgroundTasks,
                          tenant_id: str = 'default'):
    tool_node_name = f'{tool_name}_{tenant_id}'
    tool_register = ToolRegisterInfo(
        node_name=tool_node_name,
        tool_name=tool_name,
        tenant_id=tenant_id,
        workspace_dir=os.getcwd(),
    )
    background_tasks.add_task(remove_docker_container_and_update_status,
                              tool_register, app)

    return {
        'tool_node_name': tool_node_name,
        'status': ContainerStatus.exited.value
    }


@app.get('/tools/', response_model=list[ToolInstance])
async def list_tools(tenant_id: str = 'default'):
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.tenant_id == tenant_id)
        tools = session.exec(statement)
    return tools


@app.post('/tool_info/')
async def get_tool_info(tool_input: ExecuteTool):

    # get tool instance
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.name ==  # noqa W504
            f'{tool_input.tool_name}_{tool_input.tenant_id}')
        results = session.exec(statement)
        tool_instance = results.first()
    if not tool_instance:
        raise HTTPException(status_code=404, detail='Tool not found')

    # get tool service url
    try:
        tool_info_url = 'http://' + tool_instance.ip + ':' + str(
            tool_instance.port) + '/tool_info'
        response = requests.get(tool_info_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=
            f'Failed to execute tool for {tool_input.tool_name}_{tool_input.tenant_id}, with error {e}'
        )


@app.post('/execute_tool/')
async def execute_tool(tool_input: ExecuteTool):

    # get tool instance
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.name ==  # noqa W504
            f'{tool_input.tool_name}_{tool_input.tenant_id}')
        results = session.exec(statement)
        tool_instance = results.first()
    if not tool_instance:
        raise HTTPException(status_code=404, detail='Tool not found')

    tool_service_url = 'http://' + tool_instance.ip + ':' + str(
        tool_instance.port) + '/execute_tool'
    if tool_input.params == '':
        raise HTTPException(
            status_code=400,
            detail=
            f'The params of tool {tool_input.tool_name}_{tool_input.tenant_id} is empty.'
        )
    try:
        response = requests.post(
            tool_service_url,
            json={
                'params': tool_input.params,
                'kwargs': tool_input.kwargs
            })
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=
            f'Failed to execute tool for {tool_input.tool_name}_{tool_input.tenant_id}, with error {e}'
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='127.0.0.1', port=31511)
