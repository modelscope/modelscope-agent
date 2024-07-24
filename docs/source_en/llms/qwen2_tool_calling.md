# Best Practices for Qwen2 Tool Calling Service

## Table of Contents
  - [Environment Setup](#environment-setup)
  - [Model Preparation](#model-preparation)
  - [Service Invocation](#service-invocation)

## Environment Setup

```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent
```

## Model Preparation

Model Link:
- qwen2-7b-instruct: [https://modelscope.cn/models/qwen/Qwen2-7B-Instruct/summary](https://modelscope.cn/models/qwen/Qwen2-7B-Instruct/summary)

Model Download:

```python
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct')
```

## Service Invocation

By leveraging the capabilities of `modelscope-agent-server`, users can set up a local `chat/completions` service that supports OpenAI SDK calls and endows the model with tool calling capabilities. This allows models that originally only support prompt calling to quickly perform tool calling through the modelscope service.

### Service Startup

For specific usage, refer to vllm. The original command to start the `qwen2-7b-instruct` model using vllm is as follows:

```shell
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model path/to/weights
```

Now, in the modelscope-agent project directory, enter the following command to start the `tool calling` service supported by the modelscope-agent core:

```shell
sh scripts/run_assistant_server.sh --served-model-name Qwen2-7B-Instruct --model path/to/weights
```

The related service will be started on the default port **31512** and can be accessed via `http://localhost:31512`.

### Service Curl Invocation
Meanwhile, once the service is started, you can use the following method to `curl` and call the service with tool information.

```shell
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
    "model": "Qwen2-7B-Instruct",
    "messages": [
        {"content": "海淀区天气", "role": "user"}
    ]
}'
```
Return the following result:
```json
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
  "model": "Qwen2-7B-Instruct",
  "system_fingerprint": "chatcmpl_3f020464-e98d-4c7b-8717-9fca56784fe6",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 237,
    "completion_tokens": 48,
    "total_tokens": 285
  }
}
```

As you can see, using modelscope-agent-server, users can quickly enable chat models that previously could not perform tool calling to start making calls, facilitating subsequent tasks and operations.

### OpenAI SDK Invocation

Additionally, users can also make calls using the OpenAI SDK. The specific usage is as follows:

```python
from openai import OpenAI
api_base = "http://localhost:31512/v1/"
model = 'Qwen2-7B-Instruct'

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
