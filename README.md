<h1> ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a> ｜ <a href="https://arxiv.org/abs/2309.00986">Paper</a> ｜ <a href="https://modelscope.cn/studios/damo/ModelScopeGPT/summary">Demo</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Introduction

Modelscope-Agent is a customizable and scalable Agent framework. A single agent has abilities such as role-playing, LLM calling, tool usage, planning, and memory.
It mainly has the following characteristics:

- **Simple Agent Implementation Process**: Simply specify the role instruction, LLM name, and tool name list to implement an Agent application. The framework automatically arranges workflows for tool usage, planning, and memory.
- **Rich models and tools**: The framework is equipped with rich LLM interfaces, such as Dashscope and Modelscope model interfaces, OpenAI model interfaces, etc. Built in rich tools, such as **code interpreter**, **weather query**, **text to image**, **web browsing**, etc., make it easy to customize exclusive agents.
- **Unified interface and high scalability**: The framework has clear tools and LLM registration mechanism, making it convenient for users to expand more diverse Agent applications.
- **Low coupling**: Developers can easily use built-in tools, LLM, memory, and other components without the need to bind higher-level agents.


## News
* Mar 15, 2024: Modelscope-Agent and the Agentfabric (opensource version for GPTs) is running on the production environment of [modelscope studio](https://modelscope.cn/studios/agent).
* Feb 10, 2024: In Chinese New year, we upgrade the modelscope agent to version v0.3 to facilitate developers to customize various types of agents more conveniently through coding and make it easier to make multi-agent demos. For more details, you can refer to [#267](https://github.com/modelscope/modelscope-agent/pull/267) and [#293](https://github.com/modelscope/modelscope-agent/pull/293) .
* Nov 26, 2023: [AgentFabric](https://github.com/modelscope/modelscope-agent/tree/master/apps/agentfabric) now supports collaborative use in ModelScope's [Creation Space](https://modelscope.cn/studios/modelscope/AgentFabric/summary), allowing for the sharing of custom applications in the Creation Space. The update also includes the latest [GTE](https://modelscope.cn/models/damo/nlp_gte_sentence-embedding_chinese-base/summary) text embedding integration.
* Nov 17, 2023: [AgentFabric](https://github.com/modelscope/modelscope-agent/tree/master/apps/agentfabric) released, which is an interactive framework to facilitate creation of agents tailored to various real-world applications.
* Oct 30, 2023: [Facechain Agent](https://modelscope.cn/studios/CVstudio/facechain_agent_studio/summary) released a local version of the Facechain Agent that can be run locally. For detailed usage instructions, please refer to [Facechain Agent](#facechain-agent).
* Oct 25, 2023: [Story Agent](https://modelscope.cn/studios/damo/story_agent/summary) released a local version of the Story Agent for generating storybook illustrations. It can be run locally. For detailed usage instructions, please refer to [Story Agent](#story-agent).
* Sep 20, 2023: [ModelScope GPT](https://modelscope.cn/studios/damo/ModelScopeGPT/summary) offers a local version through gradio that can be run locally. You can navigate to the demo/msgpt/ directory and execute `bash run_msgpt.sh`.
* Sep 4, 2023: Three demos, [demo_qwen](demo/demo_qwen_agent.ipynb), [demo_retrieval_agent](demo/demo_retrieval_agent.ipynb) and [demo_register_tool](demo/demo_register_new_tool.ipynb), have been added, along with detailed tutorials provided.
* Sep 2, 2023: The [preprint paper](https://arxiv.org/abs/2309.00986) associated with this project was published.
* Aug 22, 2023: Support accessing various AI model APIs using ModelScope tokens.
* Aug 7, 2023: The initial version of the modelscope-agent repository was released.


## Installation

clone repo and install dependency：
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent && pip install -r requirements.txt
```

### ModelScope notebook【recommended】

The ModelScope Notebook offers a free-tier that allows ModelScope user to run the FaceChain application with minimum setup, refer to [ModelScope Notebook](https://modelscope.cn/my/mynotebook/preset)

```shell
# Step1: 我的notebook -> PAI-DSW -> GPU环境

# Step2: Download the [demo file](https://github.com/modelscope/modelscope-agent/blob/master/demo/demo_qwen_agent.ipynb) and upload it to the GPU.

# Step3:  Execute the demo notebook in order.
```


## Quickstart

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

Result
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
An example of a custom tool is provided in [demo_register_new_tool](../demo/demo_register_new_tool.ipynb)

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


## Related Tutorials

If you would like to learn more about the practical details of Agent, you can refer to our articles and video tutorials:

* [Article Tutorial](https://mp.weixin.qq.com/s/L3GiV2QHeybhVZSg_g_JRw)
* [Video Tutorial](https://b23.tv/AGIzmHM)

## Share Your Agent

We appreciate your enthusiasm in participating in our open-source ModelScope-Agent project. If you encounter any issues, please feel free to report them to us. If you have built a new Agent demo and are ready to share your work with us, please create a pull request at any time! If you need any further assistance, please contact us via email at [contact@modelscope.cn](mailto:contact@modelscope.cn) or [communication group](https://modelscope.cn/docs/%E8%81%94%E7%B3%BB%E6%88%91%E4%BB%AC)!

### Facechain Agent
Facechain is an open-source project for generating personalized portraits in various styles using facial images uploaded by users. By integrating the capabilities of Facechain into the modelscope-agent framework, we have greatly simplified the usage process. The generation of personalized portraits can now be done through dialogue with the Facechain Agent.

FaceChainAgent Studio Application Link: https://modelscope.cn/studios/CVstudio/facechain_agent_studio/summary

You can run it directly in a notebook/Colab/local environment: https://www.modelscope.cn/my/mynotebook

```
! git clone -b feat/facechain_agent https://github.com/modelscope/modelscope-agent.git

! cd modelscope-agent && ! pip install -r requirements.txt
! cd modelscope-agent/demo/facechain_agent/demo/facechain_agent && ! pip install -r requirements.txt
! pip install http://dashscope-cn-beijing.oss-cn-beijing.aliyuncs.com/zhicheng/modelscope_agent-0.1.0-py3-none-any.whl
! PYTHONPATH=/mnt/workspace/modelscope-agent/demo/facechain_agent && cd modelscope-agent/demo/facechain_agent/demo/facechain_agent && python app_v1.0.py
```
