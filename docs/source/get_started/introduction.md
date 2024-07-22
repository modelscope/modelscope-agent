# 介绍

**ModelScope-Agent**,一个适用于现实世界应用的通用且可定制的agent框架，基于开源的大型语言模型（LLMs）作为控制器。它提供了一个用户友好的系统库，具体如下：
- **可自定义引擎**: 可定制的引擎设计，支持在多个开源大型语言模型（LLMs）上进行模型训练。
- **多样化和全面的APIs**: 实现了以统一的方式与模型API和常用API的无缝集成。

为了使大型语言模型（LLMs）具备使用工具的能力，已经提出了一个全面的框架，涵盖了工具使用数据收集、工具检索、工具注册、内存控制、定制模型训练以及针对实际现实世界应用的评估。

## 怎样开始

该agent包括一个大型语言模型（LLM）以及特定任务的工具，并使用LLM来确定为了完成用户任务应该调用哪一个或哪些工具。

首先，你需要做的就是用相应的任务初始化一个`RolePlay`对象。

- 这段示例代码使用了qwen-max模型、绘图工具和天气预报工具。
    - 使用qwen-max模型需要将示例中的YOUR_DASHSCOPE_API_KEY替换为您的API-KEY，代码才能正常运行。YOUR_DASHSCOPE_API_KEY可以在[这里](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)获取。绘图工具也调用了DASHSCOPE API（wanx），因此不需要额外配置
    - 在使用天气预报工具时，您需要将示例中的YOUR_AMAP_TOKEN替换为您的高德天气API-KEY，以便代码能够正常运行。您可以在[这里](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather)获取YOUR_AMAP_TOKEN。

```Python
# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY
os.environ['AMAP_TOKEN']=YOUR_AMAP_TOKEN

# 选用RolePlay 配置agent
from modelscope_agent.agents.role_play import RolePlay

role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

# 输入工具名称
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

一个`Agent`对象由以下组件构成：

- `LLM`: 一个大型语言模型，负责处理您的输入并决定调用工具。
- `function_list`: 一个包含agent可用工具的列表。

目前，`Agent`的配置可能包含以下参数：
- `llm`: 这个agent的llm配置
    - 当为字典时：将llm的配置设置为{'model': '', 'api_key': '', 'model_server': ''}
    - 当为BaseChatModel时：llm由另一个agent发送
- `function_list`: 一个工具列表
    - 当为str时：工具名称
    - 当为Dict时：工具配置
- `storage_path`: 若无特殊指定，所有数据将在此以键值对（KV pairs）形式存储于内存中
- `instruction`: 这个agent的系统指令
- `name`: 这个agent的名称
- `description`: 用于多智能体系统的agent描述
- `kwargs`： 其他可能的参数

`Agent`作为一个基础类，不能直接被初始化和调用。需要由Agent的子类对其进行继承。子类必须实现函数`_run`，该函数主要包括三部分：消息/提示的生成、llm的调用以及基于llm结果的工具调用。我们为用户在`RolePlay`中提供了这些组件的实现，同时你也可以根据需求自定义你的组件。

```python
from modelscope_agent import Agent
class YourCustomAgent(Agent):
    def _run(self, user_request, **kwargs):
        # 定制你的工作流
```


### LLM
LLM是Agent的核心模块，它确保了交互结果的质量。
目前，Agent的配置可能包含以下参数：
- `model`: 具体的模型名称将直接传递给模型服务提供商。
- `model_server`: 模型服务的提供者

`BaseChatModel`作为llm的基础类，不能直接被初始化和调用，需要由其子类进行继承，子类必须实现函数`_chat_stream`和`_chat_no_stream`，分别对应流式输出和非流式输出。
另外，可选择实现`chat_with_functions`和`chat_with_raw_prompt`函数，分别用于函数调用和文本补全。

当前我们提供了三个模型服务提供商的实现：dashscope（支持qwen系列模型）、zhipu（支持glm系列模型）和openai（支持所有openai api格式模型）。你可以直接使用上述服务提供商支持的模型，也可以自定义你的llm。

如需更多详情，请参阅“docs/modules/llm.md”。

### `Tool`

我们提供了一些可以在agent中配置和使用的多领域工具。

您也可以通过继承基础工具，按照预定义模式设置工具的名称、描述和参数来自定义您的工具。根据您的需求，可以实现call()函数。

在[demo_register_new_tool](../../../examples/tools/register_new_tool.ipynb)中提供了一个自定义工具的例子。

您可以将您想要使用的工具名称或配置传递给agent。
```python
# 通过工具名
function_list = ['amap_weather', 'image_gen']
bot = RolePlay(function_list=function_list, ...)

# 通过工具配置
from langchain.tools import ShellTool
function_list = [{'terminal':ShellTool()}]
bot = RolePlay(function_list=function_list, ...)

# 混合使用
function_list = ['amap_weather', {'terminal':ShellTool()}]
bot = RolePlay(function_list=function_list, ...)
```

#### Built-in tools
- `image_gen`: [Wanx 图像生成](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-wanxiang). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 需要在环境变量中进行配置。
- `code_interpreter`: [代码解释器](https://jupyter-client.readthedocs.io/en/5.2.2/api/client.html)
- `web_browser`: [网页浏览](https://python.langchain.com/docs/use_cases/web_scraping)
- `amap_weather`: [高德天气](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather). AMAP_TOKEN 需要在环境变量中进行配置。
- `wordart_texture_generation`: [艺术字纹理生成](https://help.aliyun.com/zh/dashscope/developer-reference/wordart). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 需要在环境变量中进行配置。
- `web_search`: [网页搜索](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview). []
- `qwen_vl`: [Qwen-VL 图像识别](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 需要在环境变量中进行配置。
- `style_repaint`: [字符样式重绘](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-wanxiang-style-repaint). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 需要在环境变量中进行配置。
- `image_enhancement`: [追影放大镜](https://github.com/dreamoving/Phantom). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 需要在环境变量中进行配置。
- `text-address`: [地理编码](https://www.modelscope.cn/models/iic/mgeo_geographic_elements_tagging_chinese_base/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) 需要在环境变量中进行配置。
- `speech-generation`: [语音生成](https://www.modelscope.cn/models/iic/speech_sambert-hifigan_tts_zh-cn_16k/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) 需要在环境变量中进行配置。
- `video-generation`: [视频生成](https://www.modelscope.cn/models/iic/text-to-video-synthesis/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) 需要在环境变量中进行配置。
