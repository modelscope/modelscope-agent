<h1> ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Introduction
**ModelScope-Agent**, a general and customizable agent framework for real-world applications, based on open-source LLMs as controllers. It provides a user-friendly system library that are:
- **cutomizable and comprehensive framework**: customizable engine design to spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.
- **opensourced LLMs as controllers**: support model training on multiple open-source LLMs of ModelScope Community
- **Diversified and Comprehensive APIs**: enabling seamless integration with both model APIs and common APIs in a unified way.

![image](resource/modelscope-agent.png)

To equip the LLMs with tool-use abilities, a comprehensive framework has been proposed spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.

## Installation

clone repo and install dependency：
```shell
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent && pip install -r requirements.txt
```


## Quickstart

To use modelscope-agent, all you need is to instantiate an `AgentExecutor` object, and use `run()` to execute your task. For faster agent implementation, please refer to [demo_agent](demo/demo_qwen_agent.ipynb)

```Python
import os
from modelscope.utils.config import Config
from modelscope_agent.llm import LLMFactory
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.prompt import MSPromptGenerator

# get cfg from file, refer the example in config folder
model_cfg_file = os.getenv('MODEL_CONFIG_FILE', 'config/cfg_model_template.json')
model_cfg = Config.from_file(model_cfg_file)
tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', 'config/cfg_tool_template.json')
tool_cfg = Config.from_file(tool_cfg_file)


# instantiation LLM
model_name = 'modelscope-agent-qwen-7b'
llm = LLMFactory.build_llm(model_name, model_cfg)

# prompt generator
prompt_generator = MSPromptGenerator()

# instantiation agent
agent = AgentExecutor(llm, tool_cfg, prompt_generator=prompt_generator)

```

- Single-step & Multi-step tool-use

```Python
# Single-step tool-use
agent.run('使用地址识别模型，从下面的地址中找到省市区等元素，地址：浙江杭州市江干区九堡镇三村村一区', remote=True)

# Multi-step tool-use
agent.reset()
agent.run('写一篇关于Vision Pro VR眼镜的20字宣传文案，并用女声读出来，同时生成个视频看看', remote=True)
```

<div style="display: flex;">
  <img src="resource/modelscopegpt_case_single-step.png" alt="Image 1" style="width: 45%;">
  <img src="resource/modelscopegpt_case_video-generation.png" alt="Image 2" style="width: 45%;">
</div>

- Multi-turn tool-use and knowledge-qa

```Python
# Multi-turn tool-use
agent.reset()
agent.run('写一个20字左右简短的小故事', remote=True)
agent.run('用女声念出来', remote=True)
agent.run('给这个故事配一张图', remote=True)
```

<div style="display: flex;">
  <img src="resource/modelscopegpt_case_multi-turn.png" alt="Image 1" style="width: 45%;">
  <img src="resource/modelscopegpt_case_knowledge-qa.png" alt="Image 2" style="width: 45%;">
</div>


### Main components

An `AgentExecutor` object consists of the following components:

- `LLM`: A large language model that is responsibile to process your inputs and decide calling tools.
- `tool_list`: A list consists of available tools for agents.
- `PromptGenerator`: A module integrates `prompt_template`, `user_input`, `history`, `tool_list`... into efficient prompt.
- `OutputParser`: A module to parse llm response into the tools to be invoked and the corresponding parameters

We provide default implement of these components for users, but you can also custom your components according to your requirement.


### Configuration

Some configurations, `user_token` etc are not supposed to be public, so we recommend you to use `dotenv` package and `.env` file to set these configurations.

Concretely, We provide an `.env.template` file and corresponding config files in our repo. You can easily customize the configuration by referring to the provided example, and utilize your own `.env` file to read the configuration settings.

### LLM
We offer a plug-and-play LLM for users to easily utilize. The specific model details are as follows:

* modelscope-agent-qwen-7b: [modelscope-agent-qwen-7b](https://modelscope.cn/models/damo/MSAgent-Qwen-7B/summary) is a core open-source model that drives the ModelScope-Agent framework, fine-tuned based on Qwen-7B. It can be directly downloaded for local use.
* ms_gpt: A ModelScope-Agent service deployed on [DashScope](http://dashscope.aliyun.com). No local GPU resources are required. Follow the steps below to apply for the use of ms_gpt:
    1. Apply to activate the DashScope service, go to `模型广场` -> `通义千问开源系列` -> apply for a trial of `通义千问7B`. The free quota is 100,000 tokens.
    2. Create an API-KEY in `API-kEY管理`, and configure it in the `config/.env` file.


The default LLM is `ModelScope GPT`, which is deployed in a remote server and need user token to request.

If you want to use other llm, you can inherit base class and implement `generate()` or `stream_generate()` specifically.

- `generate()`: directly return final response
- `stream_generate()`: return a generator of step response, it can be used when you deploy your application in gradio.

You can also use open-source LLM from ModelScope or Huggingface and inference locally by `LLMFactory` class. Moreover, you can finetune these models with your datasets or load your custom weights.

```Python
# local llm cfg
import os
from modelscope.utils.config import Config
from modelscope_agent.llm import LLMFactory
from modelscope_agent.agent import AgentExecutor

model_name = 'modelscope-agent-qwen-7b'
model_cfg = {
    'modelscope-agent-qwen-7b':{
        'model_id': 'damo/MSAgent-Qwen-7B',
        'model_revision': 'v1.0.2',
        'use_raw_generation_config': True,
        'custom_chat': True
    }
}

tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', 'config/cfg_tool_template.json')
tool_cfg = Config.from_file(tool_cfg_file)

llm = LLMFactory(model_name, model_cfg)
agent = AgentExecutor(llm, tool_cfg)
```



### Custom tools

We provide some default pipeline tools of multiple domain that integrates in modelscope.

Also, you can custom your tools by inheriting base tool and define names, descriptions, and parameters according to pre-defined schema. And you can implement `_local_call()` or `_remote_call()` according to your requirement. Examples of supported tool are provided below. For more detailed tool registration, please refer to [tool_doc](docs/modules/tool.md) or [too_demo](demo/demo_register_new_tool.ipynb).

- Text-to-Speech Tool

```python
from modelscope_agent.tools import ModelscopePipelineTool
from modelscope.utils.constant import Tasks
from modelscope_agent.output_wrapper import AudioWrapper

class TexttoSpeechTool(ModelscopePipelineTool):
    default_model = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
    description = '文本转语音服务，将文字转换为自然而逼真的语音，可配置男声/女声'
    name = 'modelscope_speech-generation'
    parameters: list = [{
        'name': 'input',
        'description': '要转成语音的文本',
        'required': True
    }, {
        'name': 'gender',
        'description': '用户身份',
        'required': True
    }]
    task = Tasks.text_to_speech

    def _remote_parse_input(self, *args, **kwargs):
        if 'gender' not in kwargs:
            kwargs['gender'] = 'man'
        voice = 'zhibei_emo' if kwargs['gender'] == 'man' else 'zhiyan_emo'
        kwargs['parameters'] = voice
        kwargs.pop('gender')
        return kwargs

    def _parse_output(self, origin_result, remote=True):

        audio = origin_result['output_wav']
        return {'result': AudioWrapper(audio)}
```

- Text-Address Tool

```python
from modelscope_agent.tools import ModelscopePipelineTool
from modelscope.utils.constant import Tasks

class TextAddressTool(ModelscopePipelineTool):
    default_model = 'damo/mgeo_geographic_elements_tagging_chinese_base'
    description = '地址解析服务，针对中文地址信息，识别出里面的元素，包括省、市、区、镇、社区、道路、路号、POI、楼栋号、户室号等'
    name = 'modelscope_text-address'
    parameters: list = [{
        'name': 'input',
        'description': '用户输入的地址信息',
        'required': True
    }]
    task = Tasks.token_classification

    def _parse_output(self, origin_result, *args, **kwargs):
        final_result = {}
        for e in origin_result['output']:
            final_result[e['type']] = e['span']
        return final_result
```

Moreover, if the tool is a `langchain tool`, you can directly use our `LangchainTool` to wrap and adapt with current frameworks.

```Python

from modelscope_agent.tools import LangchainTool
from langchain.tools import ShellTool, ReadFileTool

# wrap langchain tools
shell_tool = LangchainTool(ShellTool())

print(shell_tool(commands=["echo 'Hello World!'", "ls"]))

```


## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@misc{modelscope-agent,
      title={ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models},
      howpublished = {\url{https://github.com/ModelScope/modelscope-agent}},
      year={2023}
}
```
