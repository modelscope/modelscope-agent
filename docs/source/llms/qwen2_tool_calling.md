# Qwen2 工具调用服务最佳实践


## 目录
- [环境准备](#环境准备)
- [模型准备](#模型准备)
- [服务调用](#服务调用)


## 环境准备
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent
```

## 模型准备
模型链接:
- qwen2-7b-instruct: [https://modelscope.cn/models/qwen/Qwen2-7B-Instruct/summary](https://modelscope.cn/models/qwen/Qwen2-7B-Instruct/summary)

模型下载:
```python
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct')
```

## 服务调用

利用`modelscope-agent-server`的能力，允许用户在本地拉起一个支持openai SDK调用的`chat/completions`服务，并且赋予该模型tool calling
的能力。 这样子可以让原本仅支持prompt调用的模型，可以通过modelscope的服务快速进行tool calling的调用。

### 服务拉起
具体使用方式参考vllm即可，原本用vllm拉起 `qwen2-7b-instruct` 模型的命令如下：
```shell
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model path/to/weights
```

现在,在modelscope-agent项目目录底下输入以下命令即可拉起由modelscope-agent内核支持的`tool calling`服务：
```shell
sh scripts/run_assistant_server.sh --served-model-name Qwen2-7B-Instruct --model path/to/weights
```
相关服务会在默认的**31512**端口上启动，可以通过`http://localhost:31512`进行访问。

### 服务curl调用
于此同时， 服务启动以后，可以通过以下方式`curl` 使用带有tool的信息调用服务。
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

返回如下结果：
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

可以看到通过modelscope-agent-server, 用户可以快速将原本无法使用tool calling的chat模型，快速开始进行调用，从而进行后续工作。


### openai SDK调用

另外，用户也可以使用openai SDK进行调用，具体使用方式如下：
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
