# Modelscope Agent Servers

Modelscope Agent Servers is a set of services that provide running environment and apis for agents, tools,
and the manager of the tool instances.

Three services are provided in this project:
- assistant server: provide chat/assistant service for agents
- tool manager server: provide apis for creating, managing, and removing tool instances
- tool node server: provide running environment for tool instances

`Assistant server` could run independently, developers could use the chat api or assistant api to interact with the agents without running tool manager service.

`Tool manager server` is bonded with `Tool node server`, the tool manager will automatically create tool node service
for each tool instance, and the tool node service will be running in a docker container with independent environment and port.
By using `tool services`, user could run tool in a more secure and stable way.


## Assistant Service

Assistant service is responsible for providing chat api for agents, two different level apis are provided in this service:
- chat: user could chat with the agent by sending `query` and `tools' info`, and the agent will respond with which tool to use and parameters needed, this api is an alternative to the LLMs who has no function call or function call result is not valid.
- assistant: user could chat with the agent by sending `query`, `tools' info`, `knowledge` and `message history`, and the agent will respond with the result of the action of the tool calling based on input.

Other than those two main apis, a file upload api is also provided for uploading files for knowledge retrieval.
The assistant service apis are running on port `31512` by default.

### Pre-requisite
- Modelscope-agent 0.5.0
- Python 3.8 or above

### Installation

```bash
# get latest code
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent

# start the assistant server
sh scripts/run_assistant_server.sh

# start the assistant server with specified backend
sh scripts/run_assistant_server.sh --model-server dashscope
```

# start the assistant server with specified model as vllm
sh scripts/run_assistant_server.sh --served-model-name Qwen2-1.5B-Instruct --model path/to/weights

### Use case

#### Chat

We provide compatibility with parts of the OpenAI API `chat/completions`, especially function calls. The developers can use `OpenAI` SDK with specified local url. Currently the supported model server includes `dashscope`, `openai` and `ollama`.

Here is an code snippet using `OpenAI` SDK with `dashscope` model server:

```Python
from openai import OpenAI
api_base = "http://localhost:31512/v1/"
model = 'qwen-max'

tools = [{
    "type": "function",
    "function": {
        "name": "amap_weather",
        "description": "amap weather tool",
        "parameters": [{
            "name": "location",
            "type": "string",
            "description": "城市/区具体名称，如`北京市海淀区`请描述为`海淀区`",
            "required": True
        }]
    }
}]

tool_choice = 'auto'

client = OpenAI(
    base_url=api_base,
    api_key="empty",

)
chat_completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": "海淀区天气是什么？"
    }],
    model=model,
    tools=tools,
    tool_choice=tool_choice
)

```

You can also use `curl` to request this API.

```Shell
curl -X POST 'http://localhost:31512/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-d '{
    "tools": [{
        "type": "function",
        "function": {
            "name": "amap_weather",
            "description": "amap weather tool",
            "parameters": [{
                "name": "location",
                "type": "string",
                "description": "城市/区具体名称，如`北京市海淀区`请描述为`海淀区`",
                "required": true
            }]
        }
    }],
    "tool_choice": "auto",
    "model": "qwen-max",
    "messages": [
        {"content": "海淀区天气", "role": "user"}
    ]
}'

```

With above examples, the output should be like this:
```Python
{
    "request_id":"xxx",
    "id":"xxx",
    "choices":[{
        "index":0,
        "message":{
            "role":"assistant",
            "content":"Action: amap_weather\nAction Input: {\"location\": \"海淀区\"}\n",
            "tool_calls":[{
                    "type":"function",
                    "function":{
                        "name":"amap_weather",
                        "arguments":"{\"location\": \"海淀区\"}"
                    }
                }]
        },
        "finish_reason":"tool_calls"
        }],
        "created":xxx,
        "model":"qwen-max",
        "object":"chat.completion",
        "usage":{"prompt_tokens":267,"completion_tokens":15,"total_tokens":282}}
```
We also support the `parallel_tool_calls` ability, the `parallel_tool_calls` are default on, for the parallel tool calling scenarios, the query and output would be

```Shell
curl -X POST 'http://localhost:31512/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-d '{
    "tools": [{
        "type": "function",
        "function": {
            "name": "amap_weather",
            "description": "amap weather tool",
            "parameters": [{
                "name": "location",
                "type": "string",
                "description": "城市/区具体名称，如`北京市海淀区`请描述为`海淀区`",
                "required": true
            }]
        }
    }],
    "tool_choice": "auto",
    "model": "qwen-max",
    "messages": [
        {"content": "请同时调用工具查找北京和上海的天气", "role": "user"}
    ]
}'

```

With above examples, the output should be like this:
```Python
{
  "request_id": "chatcmpl_058fd645-4a7a-41ca-a1db-29c9330814d6",
  "message": "",
  "output": null,
  "id": "chatcmpl_058fd645-4a7a-41ca-a1db-29c9330814d6",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Action: amap_weather\nAction Input: {\"location\": \"北京\"}\nAction: amap_weather\nAction Input: {\"location\": \"上海\"}\n\n",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "amap_weather",
              "arguments": "{\"location\": \"北京\"}"
            }
          },
          {
            "type": "function",
            "function": {
              "name": "amap_weather",
              "arguments": "{\"location\": \"上海\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "created": 1721123488,
  "model": "Qwen2-7B-Instruct",
  "system_fingerprint": "chatcmpl_058fd645-4a7a-41ca-a1db-29c9330814d6",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 333,
    "completion_tokens": 33,
    "total_tokens": 366
  }
}
```
meanwhile if you set `parallel_tool_calls` as false, then you get only tool.
```shell
curl -v -X POST 'http://localhost:31512/v1/chat/completions' -H 'Content-Type: application/json'  -d '{
    "tools": [{
        "type": "function",
        "function": {
            "name": "amap_weather",
            "description": "amap weather tool",
            "parameters": [{
                "name": "location",
                "type": "string",
                "description": "城市/区具体名称，如`北京市海淀区`请描述为`海淀区`",
                "required": true
            }]
        }
    }],
    "model": "Qwen2-7B-Instruct",
    "messages": [
        {"content": "请同时调用工具查找北京和上海的天气", "role": "user"}
    ],
    "parallel_tool_calls": false
}'
```
From the result, only one tool generated.

```json
{
  "request_id": "chatcmpl_57255f07-3d86-4b64-82e8-2a99d5f763cb",
  "message": "",
  "output": null,
  "id": "chatcmpl_57255f07-3d86-4b64-82e8-2a99d5f763cb",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Action: amap_weather\nAction Input: {\"location\": \"北京\"}\n",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "amap_weather",
              "arguments": "{\"location\": \"北京\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "created": 1721127977,
  "model": "Qwen2-7B-Instruct",
  "system_fingerprint": "chatcmpl_57255f07-3d86-4b64-82e8-2a99d5f763cb",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 246,
    "completion_tokens": 18,
    "total_tokens": 264
  }
}
```


#### Chat with vllm

User could also use the chat api with the same manner of vllm, by passing `--served-model-name` and `--model`.

An using case is shown below.

```shell
sh scripts/run_assistant_server.sh --served-model-name Qwen2-1.5B-Instruct --model /path/to/Qwen2-1___5B-Instruct
```

Then you could use `curl` to request this API or call python api as shown before

```Shell
curl -X POST 'http://localhost:31512/v1/chat/completions' \
-H 'Content-Type: application/json' \
-d '{
    "tools": [{
        "type": "function",
        "function": {
            "name": "amap_weather",
            "description": "amap weather tool",
            "parameters": [{
                "name": "location",
                "type": "string",
                "description": "城市/区具体名称，如`北京市海淀区`请描述为`海淀区`",
                "required": true
            }]
        }
    }],
    "tool_choice": "auto",
    "model": "Qwen2-1.5B-Instruct",
    "messages": [
        {"content": "海淀区天气", "role": "user"}
    ]
}'

```

With above examples, the output should be like this:
```shell
{
  "request_id": "chatcmpl_3f020464-e98d-4c7b-8717-9fca56784fe6",
  "message": "",
  "output": null,
  "id": "chatcmpl_3f020464-e98d-4c7b-8717-9fca56784fe6",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "好的，我已经调用了amap_weather工具查询了海淀区的天气情况。现在，让我为您展示一下查询结果吧。\n\n工具调用\nAction: amap_weather\nAction Input: {\"location\": \"海淀区\"}\n",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "amap_weather",
              "arguments": "{\"location\": \"海淀区\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "created": 1717485704,
  "model": "Qwen2-1.5B-Instruct",
  "system_fingerprint": "chatcmpl_3f020464-e98d-4c7b-8717-9fca56784fe6",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 237,
    "completion_tokens": 48,
    "total_tokens": 285
  }
}
```



#### Assistant

To interact with the chat API, you should construct a object like `AgentRequest` on the client side, and then use the requests library to send it as the request body.

#### knowledge retrieval

In `assistants/lite` API, to enable knowledge retrieval, you'll need to include use_knowledge and files in your configuration settings.

- `use_knowledge`: Specifies whether knowledge retrieval should be activated.
- `files`: the file(s) you wish to use during the conversation. By default, all previously uploaded files will be used.

```Python
import os
import requests

url = 'http://localhost:31512/v1/assistants/lite'

# llm config
llm_cfg = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.environ.get('DASHSCOPE_API_KEY'),
}
# agent config
agent_cfg = {
    'name': 'test',
    'description': 'test assistant',
    'instruction': 'you are a helpful assistant'
}

#
request = {
    'agent_config': agent_cfg,
    'llm_config': llm_cfg,
    'messages': [
        {'content': '高德天气API申请', 'role': 'user'}
    ],
    'uuid_str': 'test',
    'use_knowledge': True, # whether to use knowledge
    "files": ["QA.pdf"]
}

response = requests.post(url, json=request)

```

If you want to use `stream` output, you can extract messages like this:

```Python
request = {
    'agent_config': agent_cfg,
    'llm_config': llm_cfg,
    'messages': [
        {'content': '高德天气API申请', 'role': 'user'}
    ],
    'uuid_str': 'test',
    'stream': True, # whether to use stream
    'use_knowledge': True
}

response = requests.post(url, json=request)

# extract message
if response.encoding is None:
    response.encoding = 'utf-8'

for line in response.iter_lines(decode_unicode=True):
    if line:
        print(line)
```



#### Upload files
You can upload your local files use `requests` library. Below is a simple example:

```Python
import requests
# you may need to replace with your sever url
url1 = 'http://localhost:31512/v1/files'

# uuid
datas = {
    'uuid_str': 'test'
}

# the files to upload
files = [
    ('files', ('ms.txt', open('ms.txt', 'rb'), 'text/plain'))
]
response = requests.post(url1, data=datas, files=files)

```


## Tool Service

Tool service provide running environment for tool instances by docker container,
meanwhile a tool manager is responsible for creating and managing those tool instances.

User could run the agent with the tool in running in docker container, instead of same process with agent.
In such way, the tool could be run in an isolated environment, and the agent could be more stable and secure.

At the initial phase, agent will call the tool manager service to create a tool instance as tool node service, and the tool node service will
be running in a docker container with independent environment and port.

During the execution phase, agent will call the tool manager service, then tool manager will forward the call to the tool node service,
the tool manager will play as a proxy for the tool node service in this phase.

The tool manager service apis are running on port `31511` by default.
Meanwhile, the tool node services' port will be started from `31513` and increased by 1 for each new tool instance.

At last, an Oauth server will be added later to provide authentication and authorization for the tool manager service.


### Pre-requisite
- Docker installed
- Docker daemon running
- Modelscope-agent 0.5.0
- Python 3.8 or above

### Installation

```bash
# get latest code
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent

# for the first time please run the following command to install the dependencies and build tool images
sh scripts/run_tool_manager.sh build

# the rest of the time, you could just run the following command to start the tool manager service
sh scripts/run_tool_manager.sh

```

### Testing the tool service mode

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

### Running the tool service in agent

The following code snippet demonstrates how to run the tool service in agent.
1. make sure pass in the `use_tool_api=True` to the RolePlay class
2. pass in the `dashscope_api_key` to the `run` method, we will allow user to record keys in Oauth service before calling the tool manager service later, to make request much more secure.

```python
from modelscope_agent.agents.role_play import RolePlay
import os

llm_config = {'model': 'qwen-turbo', 'model_server': 'dashscope'}

# input tool name
function_list = ['image_gen']

bot = RolePlay(function_list=function_list, llm=llm_config, use_tool_api=True)

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
the only difference is that the tool service is running in a docker container by setting the `use_tool_api=True`


## Contribution of Tools
Please find detail in [tool_contribution_guide_CN.md](https://github.com/modelscope/modelscope-agent/blob/master/docs/contributing/tool_contribution_guide_CN.md)

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
    "request_id": "311f2e35-8dc3-48a3-a356-b255ee4b268c",
    "message": "",
    "output": "已完成ECS实例ID为123的续费，续费时长mon月"
}

```
