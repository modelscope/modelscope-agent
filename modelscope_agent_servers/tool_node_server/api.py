import os
from contextlib import asynccontextmanager
from typing import Coroutine
from uuid import uuid4

import json
from fastapi import FastAPI, HTTPException
from modelscope_agent.tools.base import TOOL_REGISTRY
from modelscope_agent_servers.service_utils import (create_error_msg,
                                                    create_success_msg)
from modelscope_agent_servers.tool_node_server.models import ToolRequest
from modelscope_agent_servers.tool_node_server.utils import \
    get_attribute_from_tool_cls

# Get the path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory of the current file
current_dir = os.path.dirname(current_file_path)

BASE_TOOL_DIR = os.getenv('BASE_TOOL_DIR', os.path.join(current_dir, 'assets'))
CONFIG_FILE_PATH = os.path.join(BASE_TOOL_DIR, 'configuration.json')


def get_tool_configuration(file_path: str):
    """
    Function to get the configuration of the tool.
    Configuration file is defined in the folder,

    Returns:
        dict: A dictionary containing the configuration of the tool.
        should be like:
        {
            "name": "tool_name",
            "tool_name": {
                "key": "value"
            }
        }
    """

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load tool from modelscope-agent
    """
    Startup function to initialize the required services and variables for the application.
    """
    try:
        # start docker service
        os.system('service docker start')
    except Exception:
        pass

    try:
        configs = get_tool_configuration(CONFIG_FILE_PATH)
        tool_name = configs.get('name')
        tool_cls = TOOL_REGISTRY[tool_name]
        if isinstance(tool_cls, dict):
            tool_cls = tool_cls['class']
        tool_config = configs.get(tool_name, {})
        tool_instance = tool_cls(cfg=tool_config)
        app.tool_attribute = get_attribute_from_tool_cls(tool_cls)
        app.tool_cls = tool_cls
        app.tool_instance = tool_instance
        app.tool_name = tool_name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize tool '{tool_name}' with error: {e}")
    yield


app = FastAPI(lifespan=lifespan)


@app.post('/')
@app.get('/')
async def root():
    """
    Root function that returns a message Hello World.

    Returns:
        dict: A dictionary containing a welcoming message.
    """
    request_id = str(uuid4())
    return create_success_msg({'message': 'Hello World'}, request_id)


# get tool info
@app.get('/tool_info')
async def get_tool_info(request_id: str):
    """
    Function to get the tool information.

    Returns:
        dict: A dictionary containing the tool information.
        including: name, description, parameters
    """
    try:
        tool_attribute = app.tool_attribute
        first_key = next(iter(tool_attribute))
        return create_success_msg(tool_attribute[first_key], request_id)
    except Exception as e:
        return create_error_msg(
            status_code=400,
            request_id=request_id,
            message=
            f"Failed to get tool info for '{app.tool_name}' with error {e}")


# execute tool
@app.post('/execute_tool')
async def execute_tool(request: ToolRequest):
    """
    Function to call the tool with the given tool name and request.
    Args:
        request: currently it is a string, but it should be a message later

    Returns:
    """
    tool_instance = app.tool_instance
    # call tool
    try:
        result = tool_instance.call(request.params, **request.kwargs)
        if isinstance(result, Coroutine):
            result = await result
        return create_success_msg(result, request_id=request.request_id)
    except Exception as e:
        import traceback
        print(
            f'The error is {e}, and the traceback is {traceback.format_exc()}')
        return create_error_msg(
            status_code=400,
            request_id=request.request_id,
            message=f"Failed to execute tool '{app.tool_name}' with error {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='127.0.0.1', port=31513)
