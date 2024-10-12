import os
from contextlib import asynccontextmanager
from typing import List, Optional
from uuid import uuid4

import json
import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from modelscope_agent.constants import MODELSCOPE_AGENT_TOKEN_HEADER_NAME
from modelscope_agent.tools.utils.openapi_utils import execute_api_call
from modelscope_agent_servers.service_utils import (create_error_msg,
                                                    create_success_msg,
                                                    parse_service_response)
from modelscope_agent_servers.tool_manager_server.connections import (
    create_db_and_tables, engine)
from modelscope_agent_servers.tool_manager_server.models import (
    ContainerStatus, CreateTool, ExecuteOpenAPISchema, ExecuteTool,
    ToolInstance, ToolRegisterInfo)
from modelscope_agent_servers.tool_manager_server.sandbox import (
    NODE_NETWORK, remove_docker_container, restart_docker_container,
    start_docker_container)
from modelscope_agent_servers.tool_manager_server.utils import PortGenerator
from sqlmodel import Session, select


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


# Dependency to extract the authentication token
def get_auth_token(authorization: Optional[str] = Header(
    None)) -> Optional[str]:  # noqa E125
    if authorization:
        schema, _, token = authorization.partition(' ')
        if schema.lower() == 'bearer' and token:
            # Assuming the token is a bearer token, return the token part
            return token
        elif token == '':
            return authorization

    # If the schema is not bearer or there is no token, raise an exception
    raise HTTPException(status_code=403, detail='Invalid authentication')


def get_user_token(authorization: Optional[str] = Header(
    None, alias=MODELSCOPE_AGENT_TOKEN_HEADER_NAME)):  # noqa E125
    if authorization:
        # Assuming the token is a bearer token
        schema, _, token = authorization.partition(' ')
        if schema and token and schema.lower() == 'bearer':
            return token
        elif token == '':
            return authorization
    else:
        return ''


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
    request_id = str(uuid4())

    return create_success_msg({'message': 'Hello World'},
                              request_id=request_id)


@app.post('/create_tool_service/')
async def create_tool_service(tool_info: CreateTool,
                              background_tasks: BackgroundTasks,
                              auth_token: str = Depends(get_auth_token)):
    # todo: the tool name might be the repo dir for the tool, need to parse in this situation.
    tool_node_name = f'{tool_info.tool_name}_{tool_info.tenant_id}'
    tool_register_info = ToolRegisterInfo(
        node_name=tool_node_name,
        tool_name=tool_info.tool_name,
        config=tool_info.tool_cfg,
        tenant_id=tool_info.tenant_id,
        image=tool_info.tool_image,
        workspace_dir=os.getcwd(),
        tool_url=tool_info.tool_url,
    )
    background_tasks.add_task(start_docker_container_and_store_status,
                              tool_register_info, app)

    output = {
        'tool_node_name': tool_node_name,
        'status': ContainerStatus.pending.value
    }
    request_id = str(uuid4())

    return create_success_msg(output, request_id=request_id)


@app.get('/check_tool_service_status/')
async def check_tool_service_status(tool_name: str,
                                    tenant_id: str = 'default',
                                    auth_token: str = Depends(get_auth_token)):
    # todo: the tool name might be the repo dir for the tool, need to parse in this situation.
    tool_node_name = f'{tool_name}_{tenant_id}'
    request_id = str(uuid4())

    with Session(engine) as session:
        tool_container = session.exec(
            select(ToolInstance).where(
                ToolInstance.name == tool_node_name)).first()
        if not tool_container:
            return create_error_msg(
                status_code=404,
                request_id=request_id,
                message='Tool not found')
        output = {'status': tool_container.status}
        message = ''
        if tool_container.status == ContainerStatus.failed.value:
            message = tool_container.error
        result = create_success_msg(
            output, request_id=request_id, message=message)
        return result


@app.post('/update_tool_service/')
async def update_tool_service(tool_info: CreateTool,
                              background_tasks: BackgroundTasks,
                              auth_token: str = Depends(get_auth_token)):
    tool_node_name = f'{tool_info.tool_name}_{tool_info.tenant_id}'
    tool_register_info = ToolRegisterInfo(
        node_name=tool_node_name,
        tool_name=tool_info.tool_name,
        config=tool_info.tool_cfg,
        tenant_id=tool_info.tenant_id,
        image=tool_info.tool_image,
        workspace_dir=os.getcwd(),
        tool_url=tool_info.tool_url,
    )
    background_tasks.add_task(restart_docker_container_and_update_status,
                              tool_register_info, app)

    output = {
        'tool_node_name': tool_node_name,
        'status': ContainerStatus.pending.value
    }
    request_id = str(uuid4())

    return create_success_msg(output, request_id=request_id)


@app.post('/remove_tool/')
async def deregister_tool(tool_name: str,
                          background_tasks: BackgroundTasks,
                          tenant_id: str = 'default',
                          auth_token: str = Depends(get_auth_token)):
    tool_node_name = f'{tool_name}_{tenant_id}'
    tool_register = ToolRegisterInfo(
        node_name=tool_node_name,
        tool_name=tool_name,
        tenant_id=tenant_id,
        workspace_dir=os.getcwd(),
    )
    background_tasks.add_task(remove_docker_container_and_update_status,
                              tool_register, app)

    output = {
        'tool_node_name': tool_node_name,
        'status': ContainerStatus.exited.value
    }
    request_id = str(uuid4())

    return create_success_msg(output, request_id=request_id)


@app.get('/tools/', response_model=List[ToolInstance])
async def list_tools(tenant_id: str = 'default',
                     auth_token: str = Depends(get_auth_token)):
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.tenant_id == tenant_id)
        tools = session.exec(statement)
    request_id = str(uuid4())

    return create_success_msg({'tools': tools}, request_id=request_id)


@app.post('/tool_info/')
async def get_tool_info(tool_input: ExecuteTool,
                        user_token: str = Depends(get_user_token),
                        auth_token: str = Depends(get_auth_token)):

    # get tool instance
    request_id = str(uuid4())

    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.name ==  # noqa W504
            f'{tool_input.tool_name}_{tool_input.tenant_id}')
        results = session.exec(statement)
        tool_instance = results.first()
    if not tool_instance:
        return create_error_msg(
            status_code=404, request_id=request_id, message='Tool not found')

    # get tool service url
    try:
        tool_info_url = 'http://' + tool_instance.ip + ':' + str(
            tool_instance.port) + '/tool_info'
        response = requests.get(
            tool_info_url,
            params={'request_id': request_id},
            headers={'Authorization': f'Bearer {user_token}'})
        response.raise_for_status()
        return create_success_msg(
            parse_service_response(response), request_id=request_id)
    except Exception as e:
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=
            f'Failed to get tool info for {tool_input.tool_name}_{tool_input.tenant_id}, with error {e}'
        )


@app.post('/execute_tool/')
async def execute_tool(tool_input: ExecuteTool,
                       user_token: str = Depends(get_user_token),
                       auth_token: str = Depends(get_auth_token)):

    request_id = str(uuid4())

    # get tool instance
    with Session(engine) as session:
        statement = select(ToolInstance).where(
            ToolInstance.name ==  # noqa W504
            f'{tool_input.tool_name}_{tool_input.tenant_id}')
        results = session.exec(statement)
        tool_instance = results.first()
    if not tool_instance:
        return create_error_msg(
            status_code=404, request_id=request_id, message='Tool not found')

    tool_service_url = 'http://' + tool_instance.ip + ':' + str(
        tool_instance.port) + '/execute_tool'
    if tool_input.params == '':
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=
            f'The params of tool {tool_input.tool_name}_{tool_input.tenant_id} is empty.'
        )
    try:
        response = requests.post(
            tool_service_url,
            json={
                'params': tool_input.params,
                'kwargs': tool_input.kwargs,
                'request_id': request_id
            },
            headers={'Authorization': f'Bearer {user_token}'})
        response.raise_for_status()
        return create_success_msg(
            parse_service_response(response), request_id=request_id)
    except Exception as e:
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=
            f'Failed to execute tool for {tool_input.tool_name}_{tool_input.tenant_id}, '
            f'with error: {e} and origin error {response.message}')


@app.post('/openapi_schema')
async def get_openapi_schema(openapi_input: ExecuteOpenAPISchema,
                             user_token: str = Depends(get_user_token),
                             auth_token: str = Depends(get_auth_token)):

    # get tool instance
    request_id = str(uuid4())

    # TODO(Zhicheng): should implement this function to get schema based on openapi schema name from database
    #  with an api for saving scheme to database
    # a fixed openapi schema is used here for demo
    openapi_schema = {
        'openapi': '3.0.1',
        'info': {
            'title': 'TODO Plugin',
            'description':
            'A plugin that allows the user to create and manage a TODO list using ChatGPT. ',
            'version': 'v1'
        },
        'servers': [{
            'url': 'http://localhost:5003'
        }],
        'paths': {
            '/todos/{username}': {
                'get': {
                    'operationId':
                    'getTodos',
                    'summary':
                    'Get the list of todos',
                    'parameters': [{
                        'in': 'path',
                        'name': 'username',
                        'schema': {
                            'type': 'string'
                        },
                        'required': True,
                        'description': 'The name of the user.'
                    }],
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref':
                                        '#/components/schemas/getTodosResponse'
                                    }
                                }
                            }
                        }
                    }
                },
                'post': {
                    'operationId':
                    'addTodo',
                    'summary':
                    'Add a todo to the list',
                    'parameters': [{
                        'in': 'path',
                        'name': 'username',
                        'schema': {
                            'type': 'string'
                        },
                        'required': True,
                        'description': 'The name of the user.'
                    }],
                    'requestBody': {
                        'required': True,
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref':
                                    '#/components/schemas/addTodoRequest'
                                }
                            }
                        }
                    },
                    'responses': {
                        '200': {
                            'description': 'OK'
                        }
                    }
                },
                'delete': {
                    'operationId':
                    'deleteTodo',
                    'summary':
                    'Delete a todo from the list',
                    'parameters': [{
                        'in': 'path',
                        'name': 'username',
                        'schema': {
                            'type': 'string'
                        },
                        'required': True,
                        'description': 'The name of the user.'
                    }],
                    'requestBody': {
                        'required': True,
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref':
                                    '#/components/schemas/deleteTodoRequest'
                                }
                            }
                        }
                    },
                    'responses': {
                        '200': {
                            'description': 'OK'
                        }
                    }
                }
            }
        },
        'components': {
            'schemas': {
                'getTodosResponse': {
                    'type': 'object',
                    'properties': {
                        'todos': {
                            'type': 'array',
                            'items': {
                                'type': 'string'
                            },
                            'description': 'The list of todos.'
                        }
                    }
                },
                'addTodoRequest': {
                    'type': 'object',
                    'required': ['todo'],
                    'properties': {
                        'todo': {
                            'type': 'string',
                            'description': 'The todo to add to the list.',
                            'required': True
                        }
                    }
                },
                'deleteTodoRequest': {
                    'type': 'object',
                    'required': ['todo_idx'],
                    'properties': {
                        'todo_idx': {
                            'type': 'integer',
                            'description': 'The index of the todo to delete.',
                            'required': True
                        }
                    }
                }
            }
        }
    }
    # get tool service url
    try:

        return create_success_msg(openapi_schema, request_id=request_id)
    except Exception as e:
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=
            f'Failed to get openapi schema for {openapi_input.openapi_name} with error {e}'
        )


@app.post('/execute_openapi')
async def execute_openapi(openapi_input: ExecuteOpenAPISchema,
                          user_token: str = Depends(get_user_token),
                          auth_token: str = Depends(get_auth_token)):

    request_id = str(uuid4())

    if openapi_input.params == '':
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=f'The params of tool {openapi_input.tool_name}is empty.')

    try:
        url = openapi_input.url
        headers = openapi_input.headers
        method = openapi_input.method.upper()
        if isinstance(openapi_input.params, str):
            params = json.loads(openapi_input.params)
        else:
            params = openapi_input.params
        data = openapi_input.data
        response = execute_api_call(url, method, headers, params, data,
                                    openapi_input.cookies)
        return create_success_msg(response, request_id=request_id)
    except Exception as e:
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=
            f'Failed to execute openapi for {openapi_input.openapi_name}, '
            f'with error: {e}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='127.0.0.1', port=31511)
