import os
from contextlib import asynccontextmanager

import json
from fastapi import Body, FastAPI, HTTPException
from modelscope_agent.tools import TOOL_REGISTRY, BaseTool

from .models import ToolRequest, ToolResponse

CONFIG_FILE_PATH = os.path.join('/app/assets', 'configuration.json')


def get_tool_configuration():
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
        with open(CONFIG_FILE_PATH, 'r') as f:
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
    configs = get_tool_configuration()
    tool_name = configs.get('name')
    try:
        tool_cls = TOOL_REGISTRY.get(tool_name, None)
        tool_config = configs.get(tool_name, {})
        tool_instance = tool_cls(**tool_config)
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
    return {'message': 'Hello World'}


# execute tool
@app.post('/execute_tool')
def call_tool(request: ToolRequest):
    """
    Function to call the tool with the given tool name and request.
    Args:
        request: currently it is a string, but it should be a message later

    Returns:
    """
    tool_instance = app.tool_instance
    # 创建工具实例，并调用它
    result = tool_instance.call(request.params)

    return result
