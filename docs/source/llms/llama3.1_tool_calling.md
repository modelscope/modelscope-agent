# Llama3.1 工具调用服务最佳实践


## 目录
- [环境准备](#环境准备)
- [模型准备](#模型准备)
- [服务调用](#服务调用)


## 环境准备
Llama3.1部署依赖vllm 最新补丁版本 0.5.3.post1

```shell
# speed up if needed
# pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple
# pip config set install.trusted-host mirrors.cloud.aliyuncs.com
pip install https://github.com/vllm-project/vllm/releases/download/v0.5.3.post1/vllm-0.5.3.post1+cu118-cp310-cp310-manylinux1_x86_64.whl
```
依赖modelscope-agent项目下的modelscope-agent-server进行tool calling能力调用
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent
```

## 模型准备
模型链接:
- meta-llama/Meta-Llama-3.1-8B-Instruct: [https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct)

模型下载:
```python
from modelscope import snapshot_download
model = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct")
```
打印 model获得model本地地址 /path/to/weights

## 服务调用

利用`modelscope-agent-server`的能力，允许用户在本地拉起一个支持openai SDK调用的`chat/completions`服务，并且赋予该模型tool calling
的能力。 这样子可以让原本仅支持prompt调用的模型，可以通过modelscope的服务快速进行tool calling的调用。

### 服务拉起
具体使用方式参考vllm即可，原本用vllm拉起 `meta-llama/Meta-Llama-3.1-8B-Instruct` 模型的命令如下：
```shell
python -m vllm.entrypoints.openai.api_server --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct --model path/to/weights
```

现在,在modelscope-agent项目目录底下输入以下命令即可拉起由modelscope-agent内核支持的`tool calling`服务：
```shell
sh scripts/run_assistant_server.sh --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct --model path/to/weights
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
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
        {"content": "海淀区天气", "role": "user"}
    ]
}'
```

返回如下结果：
```json
{
  "request_id": "chatcmpl_84a66af2-4021-4ae6-822d-8e3f42ca9f43",
  "message": "",
  "output": null,
  "id": "chatcmpl_84a66af2-4021-4ae6-822d-8e3f42ca9f43",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "工具调用\nAction: amap_weather\nAction Input: {\"location\": \"北京市\"}\n",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "amap_weather",
              "arguments": "{\"location\": \"北京市\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "created": 1721803228,
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "system_fingerprint": "chatcmpl_84a66af2-4021-4ae6-822d-8e3f42ca9f43",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
```

可以看到通过modelscope-agent-server, 用户可以快速将原本无法使用tool calling的chat模型，快速开始进行调用，从而进行后续工作。


### openai SDK调用

另外，用户也可以使用openai SDK进行调用，具体使用方式如下：
```python
from openai import OpenAI
api_base = "http://localhost:31512/v1/"
model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

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

### 70B模型Tool calling调用
对于70B的模型调用依赖4张A100的卡能够跑到llama3.1的max_model_len（131072），或者选择2张卡，可以限制模型的max_model_len=8192
具体示例如下：
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3;sh scripts/run_assistant_server.sh --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct --model '/path/to/weights' --tensor-parallel-size 4
```

或者双卡 并限制max_model_len
```shell
export CUDA_VISIBLE_DEVICES=0,1;sh scripts/run_assistant_server.sh --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct --model '/path/to/weights' --tensor-parallel-size 2 --max_model_len 8192
```
