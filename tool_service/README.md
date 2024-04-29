# Tool Service

Tool service provide running environment for tool instances by docker container,
meanwhile a tool manager is responsible for creating and managing those tool instances.

User could run the agent with the tool in running in docker container, instead of same process with agent.
In such way, the tool could be run in an isolated environment, and the agent could be more stable and secure.

At the initial phase, agent will call the tool manager service to create a tool instance as tool node service, and the tool node service will
be running in a docker container with independent environment and port.

During the execution phase, agent will call the tool manager service, then tool manager will forward the call to the tool node service,
the tool manager will play as a proxy for the tool node service in this phase.

At last, an Oauth server will be added later to provide authentication and authorization for the tool manager service.


## Pre-requisite
- Docker installed
- Docker daemon running
- Python 3.6 or above

## Installation

```bash
# get latest code
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent

# for the first time please run the following command to install the dependencies and build tool images
sh scripts/run_tool_manager.sh build

# the rest of the time, you could just run the following command to start the tool manager service
sh scripts/run_tool_manager.sh

```

## Testing the tool service mode

1. start the tool manager server by running
```bash
sh scripts/run_tool_manager.sh
```

2. init modelscope agent tool node service by calling
```python
from modelscope_agent.tools.base import ToolServiceProxy

tool_service = ToolServiceProxy(tool_name='RenewInstance', tool_cfg={'test': 'xxx'})
```

3. call the tool service by
```python
result = tool_service.call( "{\"instance_id\": 123, \"period\": \"mon\"}")
```

## Running the tool service in agent

The following code snippet demonstrates how to run the tool service in agent.
1. make sure pass in the `use_api=True` to the RolePlay class
2. pass in the `dashscope_api_key` to the `run` method, we will allow user to record keys in Oauth service before calling the tool manager service later, to make request much more secure.

```python
from modelscope_agent.agents.role_play import RolePlay
import os

llm_config = {'model': 'qwen-turbo', 'model_server': 'dashscope'}

# input tool name
function_list = ['image_gen']

bot = RolePlay(function_list=function_list, llm=llm_config, use_api=True)

response = bot.run(
    '创建一个多啦A梦', dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))

text = ''
for chunk in response:
    text += chunk
print(text)
assert isinstance(text, str)
assert 'Answer:' in text
assert 'Observation:' in text
assert '![IMAGEGEN]' in text
```
We could see that the usage of the tool service has almost no difference with the local tool service,
the only difference is that the tool service is running in a docker container by setting the `use_api=True`


## Contribution of Tools
Please find detail in

## Tool Manager API
The tool manager API is responsible for creating and managing tool container, it is running on port 31511 by default.

The API is documented by OpenAPI, you can access the API document by visiting `http://localhost:31511/docs` after starting the tool manager.

### API Endpoints

#### Create a tool instance
```
POST /create_tool_service
{
  "tool_name": "RenewInstance",
  "tenant_id": "default",
  "tool_cfg": {},
  "tool_image": "modelscope-agent/tool-node:lastest"
}

Response
{
    "status_code": 200,
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output":
        {
          "tool_node_name": "RenewInstance_default",
          "status": "pending"
        }
}

```

#### Get tool instance status
```
POST /check_tool_service_status/
{
  "tool_name": "string",
  "tenant_id": "default"
}

Response
{
    "status_code": 200,
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output":
        {
          "tool_node_name": "RenewInstance_default",
          "status": "running"
        }
}
```

#### Remove a tool instance
```
POST /remove_tool/
{
  "tool_name": "RenewInstance",
  "tenant_id": "default"
}

Response
{
    "status_code": 200,
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output":
        {
          "tool_node_name": "RenewInstance_default",
          "status": "exited"
        }
}
```

#### Get tenant tool instances
```
GET /tools/?tenant_id=default

Response
{
    "status_code": 200,
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output":
        {
          "tools":{
            "RenewInstance_default": "running",
            "CreateInstance_default": "exited"
          }
        }
}
```
#### Get tool info
```
POST /tool_info/
{
  "tool_name": "string",
  "tenant_id": "default",
}

Response
{
    "status_code": 200,
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output": {
        "description": "续费一台包年包月ECS实例",
        "name": "RenewInstance",
        "parameters"[{
            "name": "instance_id",
            "description": "ECS实例ID",
            "required": True,
            "type": "string"
        }, {
            "name": "period",
            "description": "续费时长以月为单位",
            "required": True,
            "type": "string"
        }]
    }
}
```

#### Execute tool
```
POST /execute_tool/
{
  "tool_name": "string",
  "tenant_id": "default",
  "params": "{\"instance_id\": 123, \"period\": \"mon\"}",
  "kwargs": {}

}

Response
{
    "status_code": 200,
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output": "已完成ECS实例ID为123的续费，续费时长mon月"
}

```
