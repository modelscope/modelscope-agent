# Best Practices for Llama3.1 Tool Calling Service

## Table of Contents
  - [Environment Setup](#environment-setup)
  - [Model Preparation](#model-preparation)
  - [Service Invocation](#service-invocation)

## Environment Setup
Llama3.1 depends on the latest vllm version 0.5.3.post1, please make sure install it firstly.
```shell
# speed up if needed
# pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple
# pip config set install.trusted-host mirrors.cloud.aliyuncs.com
pip install https://github.com/vllm-project/vllm/releases/download/v0.5.3.post1/vllm-0.5.3.post1+cu118-cp310-cp310-manylinux1_x86_64.whl
```

For tool calling, please make sure using the modelscope-agent-server from modelscope-agent project
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent
```

## Model Preparation

Model Link:
- meta-llama/Meta-Llama-3.1-8B-Instruct: [https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct)

Model Download:

```python
from modelscope import snapshot_download
model = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct")
```
or
```python
from huggingface_hub import snapshot_download
model = snapshot_download("meta-llama/Meta-Llama-3.1-8B-Instruct")
```
Print the `model` to get the local path to weights


## Service Invocation

By leveraging the capabilities of `modelscope-agent-server`, users can set up a local `chat/completions` service that supports OpenAI SDK calls and endows the model with tool calling capabilities. This allows models that originally only support prompt calling to quickly perform tool calling through the modelscope service.

### Service Startup

For specific usage, refer to vllm. The original command to start the `meta-llama/Meta-Llama-3.1-8B-Instruct` model using vllm is as follows:

```shell
python -m vllm.entrypoints.openai.api_server --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct --model path/to/weights
```

Now, in the modelscope-agent project directory, enter the following command to start the `tool calling` service supported by the modelscope-agent core:

```shell
sh scripts/run_assistant_server.sh --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct --model path/to/weights
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
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
        {"content": "海淀区天气", "role": "user"}
    ]
}'
```
Return the following result:
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

As you can see, using modelscope-agent-server, users can quickly enable chat models that previously could not perform tool calling to start making calls, facilitating subsequent tasks and operations.

### OpenAI SDK Invocation

Additionally, users can also make calls using the OpenAI SDK. The specific usage is as follows:

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

### 70B Tool calling Service Scripts
In order to get full ability of llama3.1 70B, 4*A100 should be ready for the max sequence length with 131072.
However, 2*A100 could run the 70B as well with limit the max sequence length to 8192.
The following scripts shows the scripts

max sequence length with 131072 and 4*A100
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3;sh scripts/run_assistant_server.sh --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct --model '/path/to/weights' --tensor-parallel-size 4
```

max sequence length with 8192 and 2*A100
```shell
export CUDA_VISIBLE_DEVICES=0,1;sh scripts/run_assistant_server.sh --served-model-name meta-llama/Meta-Llama-3.1-70B-Instruct --model '/path/to/weights' --tensor-parallel-size 2 --max_model_len 8192
```
