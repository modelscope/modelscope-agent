# Introduction

**ModelScope-Agent**, a general and customizable agent framework for real-world applications, based on open-source LLMs as controllers. It provides a user-friendly system library that are:
- **customizable engine**: customizable engine design to support model training on multiple open-source LLMs
- **Diversified and Comprehensive APIs**: enabling seamless integration with both model APIs and common APIs in a unified way.

To equip the LLMs with tool-use abilities, a comprehensive framework has been proposed spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.


## How to start

The agent incorporates an LLM along with task-specific tools, and uses the LLM to determine which tool or tools to invoke in order to complete the user's tasks.

To start, all you need to do is initialize an `RolePlay` object with corresponding tasks

- 这个示例代码使用了qwen-max模型，画图工具和天气预报工具。
    - 使用qwen-max模型需要使用您的API-KEY替换示例中的 YOUR_DASHSCOPE_API_KEY，代码才能正常运行。YOUR_DASHSCOPE_API_KEY可以在[这里](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.i0)获取。画图工具调用的也是DASHSCOPE API（wanx万象），因此不需要额外配置。
    - 使用天气预报工具需要使用您的高德天气API-KEY替换示例中的YOUR_AMAP_TOKEN，代码才能正常运行。YOUR_AMAP_TOKEN可以在[这里](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather)获取。


```Python
# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY
os.environ['AMAP_TOKEN']=YOUR_AMAP_TOKEN

# 选用RolePlay 配置agent
from modelscope_agent.agents.role_play import RolePlay  # NOQA

role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

# input tool name
function_list = ['amap_weather', 'image_gen']

bot = RolePlay(
    function_list=function_list, llm=llm_config, instruction=role_template)

response = bot.run('朝阳区天气怎样？')

text = ''
for chunk in response:
    text += chunk
```

运行结果
- terminal运行，llm输出
```shell
# 第一次调用llm的输出
Action: amap_weather
Action Input: {"location": "朝阳区"}

# 第二次调用llm的输出
目前，朝阳区的天气状况为阴天，气温为1度。

Action: image_gen
Action Input: {"text": "朝阳区城市风光", "resolution": "1024*1024"}

# 第三次调用llm的输出
目前，朝阳区的天气状况为阴天，气温为1度。同时，我已为你生成了一张朝阳区的城市风光图，如下所示：

![](https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1d/45/20240204/3ab595ad/96d55ca6-6550-4514-9013-afe0f917c7ac-1.jpg?Expires=1707123521&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=RsJRt7zsv2y4kg7D9QtQHuVkXZY%3D)
```

## modules
### Agent

An `Agent` object consists of the following components:

- `LLM`: A large language model that is responsible to process your inputs and decide calling tools. 
- `function_list`: A list consists of available tools for agents.

Currently, configuration of `Agent` may contain following arguments:
- `llm`: The llm config of this agent
    - When Dict: set the config of llm as {'model': '', 'api_key': '', 'model_server': ''}
    - When BaseChatModel: llm is sent by another agent
- `function_list`: A list of tools
    - When str: tool names
    - When Dict: tool cfg
- `storage_path`: If not specified otherwise, all data will be stored here in KV pairs by memory
- `instruction`: the system instruction of this agent
- `name`: the name of agent
- `description`: the description of agent, which is used for multi_agent
- `kwargs`: other potential parameters

`Agent`作为一个基类无法直接被初始化调用，它的`_run`函数还没有被实现。`_run`函数为Agent的运行流程，主要包括三部分：messages/propmt的生成、llm的调用、根据llm的结果进行工具调用。使用时您需要调用其实现了`_run`函数的子类。We provide an implement of these components in `RolePlay` for users, and you can also custom your components according to your requirement.

```python
from modelscope_agent import Agent
class YourCustomAgent(Agent):
    def _run(self, user_request, **kwargs):
        # Custom your workflow
```
To custom your llm, please refer to `agent.md`

### LLM
LLM is core module of agent, which ensures the quality of interaction results.

Currently, configuration of `LLM` may contain following arguments:
- `model`: 具体的模型名，将被直接传给模型服务提供商。
- `model_server`: 模型服务的提供商。

LLM subclasses need to inherit it. They must implement interfaces _chat_stream and _chat_no_stream, which correspond to streaming output and non-streaming output respectively.
Optionally implement chat_with_functions and chat_with_raw_prompt for function calling and text completion.

我们提供了dashscope（为qwen系列模型）, zhipu(为glm系列模型) and openai(为所有openai api格式的模型)三个模型服务提供商的实现。您可以直接使用上述服务商支持的模型，也可以定制您的llm。

To custom your llm, please refer to `llm.md`

### `Tool`

