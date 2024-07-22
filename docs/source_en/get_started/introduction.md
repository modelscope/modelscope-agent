# Introduction

**ModelScope-Agent**, a general and customizable agent framework for real-world applications, based on open-source LLMs as controllers. It provides a user-friendly system library that are:
- **customizable engine**: customizable engine design to support model training on multiple open-source LLMs
- **Diversified and Comprehensive APIs**: enabling seamless integration with both model APIs and common APIs in a unified way.

To equip the LLMs with tool-use abilities, a comprehensive framework has been proposed spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.


## How to start

The agent incorporates an LLM along with task-specific tools, and uses the LLM to determine which tool or tools to invoke in order to complete the user's tasks.

To start, all you need to do is initialize an `RolePlay` object with corresponding tasks

- This sample code uses the qwen-max model, drawing tools and weather forecast tools.
     - Using the qwen-max model requires replacing YOUR_DASHSCOPE_API_KEY in the example with your API-KEY for the code to run properly. YOUR_DASHSCOPE_API_KEY can be obtained [here](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key). The drawing tool also calls DASHSCOPE API (wanx), so no additional configuration is required.
     - When using the weather forecast tool, you need to replace YOUR_AMAP_TOKEN in the example with your AMAP weather API-KEY so that the code can run normally. YOUR_AMAP_TOKEN is available [here](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather).


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

result
- Terminal runs
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

`Agent`, as a base class, cannot be directly initialized and called. Agent subclasses need to inherit it. They must implement function `_run`, which mainly includes three parts: generation of messages/propmt, calling of llm(s), and tool calling based on the results of llm. We provide an implement of these components in `RolePlay` for users, and you can also custom your components according to your requirement.

```python
from modelscope_agent import Agent
class YourCustomAgent(Agent):
    def _run(self, user_request, **kwargs):
        # Custom your workflow
```


### LLM
LLM is core module of agent, which ensures the quality of interaction results.

Currently, configuration of `` may contain following arguments:
- `model`: The specific model name will be passed directly to the model service provider.
- `model_server`: provider of model services.

`BaseChatModel`, as a base class of llm, cannot be directly initialized and called. The subclasses need to inherit it. They must implement function `_chat_stream` and `_chat_no_stream`, which correspond to streaming output and non-streaming output respectively.
Optionally implement `chat_with_functions` and `chat_with_raw_prompt` for function calling and text completion.

Currently we provide the implementation of three model service providers: dashscope (for qwen series models), zhipu (for glm series models) and openai (for all openai api format models). You can directly use the models supported by the above service providers, or you can customize your llm.

For more information please refer to `docs/modules/llm.md`

### `Tool`

We provide several multi-domain tools that can be configured and used in the agent.

You can also customize your tools with set the tool's name, description, and parameters based on a predefined pattern by inheriting the base tool. Depending on your needs, call() can be implemented.
An example of a custom tool is provided in [demo_register_new_tool](../../../examples/tools/register_new_tool.ipynb)

You can pass the tool name or configuration you want to use to the agent.
```python
# by tool name
function_list = ['amap_weather', 'image_gen']
bot = RolePlay(function_list=function_list, ...)

# by tool configuration
from langchain.tools import ShellTool
function_list = [{'terminal':ShellTool()}]
bot = RolePlay(function_list=function_list, ...)

# by mixture
function_list = ['amap_weather', {'terminal':ShellTool()}]
bot = RolePlay(function_list=function_list, ...)
```

#### Built-in tools
- `image_gen`: [Wanx Image Generation](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-wanxiang). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) needs to be configured in the environment variable.
- `code_interpreter`: [Code Interpreter](https://jupyter-client.readthedocs.io/en/5.2.2/api/client.html)
- `web_browser`: [Web Browsing](https://python.langchain.com/docs/use_cases/web_scraping)
- `amap_weather`: [AMAP Weather](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather). AMAP_TOKEN needs to be configured in the environment variable.
- `wordart_texture_generation`: [Word art texture generation](https://help.aliyun.com/zh/dashscope/developer-reference/wordart). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) needs to be configured in the environment variable.
- `web_search`: [Web Searching](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview). []
- `qwen_vl`: [Qwen-VL image recognition](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) needs to be configured in the environment variable.
- `style_repaint`: [Character style redrawn](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-wanxiang-style-repaint). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) needs to be configured in the environment variable.
- `image_enhancement`: [Chasing shadow-magnifying glass](https://github.com/dreamoving/Phantom). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) needs to be configured in the environment variable.
- `text-address`: [Geocoding](https://www.modelscope.cn/models/iic/mgeo_geographic_elements_tagging_chinese_base/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) needs to be configured in the environment variable.
- `speech-generation`: [Speech generation](https://www.modelscope.cn/models/iic/speech_sambert-hifigan_tts_zh-cn_16k/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) needs to be configured in the environment variable.
- `video-generation`: [Video generation](https://www.modelscope.cn/models/iic/text-to-video-synthesis/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) needs to be configured in the environment variable.
