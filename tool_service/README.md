# Tool Service

Tool service provide running environment for tool instances by docker container,
meanwhile a tool manager part is responsible for creating and managing those tool instances.


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
  "tool_image": "modelscope-agent/tool-node:v0.4"
}

Response
{
  "tool_node_name": "RenewInstance_default",
  "status": "pending"
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
  "status": "running"
}
```

#### Remove a tool instance
```
POST /remove_tool/
{
  "tool_name": "RenewInstance",
  "tenant_id": "default"
}

RESPONSE
{
  "tool_node_name": "RenewInstance_default",
  "status": "exited"
}
```

#### Get tenant tool instances
```
GET /tools/?tenant_id=default

```

#### Get tool service url
```
POST /get_tool_service_url/
{
  "tool_name": "string",
  "tenant_id": "default"
}

RESPONSE
"http://localhost:31513/execute_tool"

```



## Tool Node API
The tool node API only support one endpoint, which is used to execute the tool with the tool name
The params is a string, which is the input for the tool instance.

```
POST /execute_tool
{
  "params": "string"
}

RESPONSE
"json string"
```

## Testing the tool service mode

1. start the tool manager server
2. init modelscope agent tool node service by calling
```python
from modelscope_agent.tools import ToolServiceProxy

tool_service = ToolServiceProxy(tool_name='RenewInstance', tool_cfg={'test': 'xxx'})
```
3. call the tool service by
```python
result = tool_service.call( "{\"instance_id\": 123, \"period\": \"mon\"}")
```

## Running the tool service in agent
