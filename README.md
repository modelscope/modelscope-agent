# ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models

**ModelScope-Agent**, a general and customizable agent framework for real-world applications, based on open-source LLMs as controllers. It provides a user-friendly system library that are:
- **cutomizable and comprehensive framework**: customizable engine design to spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.
- **opensourced LLMs as controllers**: support model training on multiple open-source LLMs of ModelScope Community
- **Diversified and Comprehensive APIs**: enabling seamless integration with both model APIs and common APIs in a unified way.

![image](resource/modelscope-agent.png)

To equip the LLMs with tool-use abilities, a comprehensive framework has been proposed spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.

## Quickstart

To use modelscope-agent, all you need is to instantiate an `AgentExecutor` object, and use `run()` to execute your task.

```Python
# instantiate llm
llm = ModelScopeGPT(model_cfg)

# instantiate agent
agent = AgentExecutor(llm, tool_cfg)

```

- Single-step tool-use

```Python
agent.run('使用地址识别模型，从下面的地址中找到省市区等元素，地址：浙江杭州市江干区九堡镇三村村一区', remote=True)
```
![image](resource/modelscopegpt_case_single-step.png)


- Multi-step tool-use

```Python
agent.run('写一篇关于Vision Pro VR眼镜的20字宣传文案，并用女声读出来，同时生成个视频看看', remote=True)
```

![image](resource/modelscopegpt_case_video-generation.png)

- Multi-turn tool-use

```Python
agent.run('写一个20字左右简短的小故事', remote=True)
agent.run('用女声念出来', remote=True)
agent.run('给这个故事配一张图', remote=True)
```
![image](resource/modelscopegpt_case_multi-turn.png)


- Multi-turn knowledge-qa

![image](resource/modelscopegpt_case_knowledge-qa.png)


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

### Custom your LLM

The default LLM is `ModelScope GPT`, which is deployed in a remote server and need user token to request.

If you want to use other llm, you can inherit base class and implement `generate()` or `stream_generate()` specifically.

- `generate()`: directly return final response
- `stream_generate()`: return a generator of step response, it can be used when you deploy your application in gradio.

You can also use open-source LLM from ModelScope or Huggingface and inference locally by `LocalLLM` class. Moreover, you can finetune these models with your datasets or load your custom weights.

```Python
# local llm cfg
model_name = 'modelscope-agent-qwen-7b'
model_cfg = {
    'modelscope-agent-qwen-7b':{
        'model_id': 'damo/MSAgent-Qwen-7B',
        'model_revision': 'v1.0.1',
        'use_raw_generation_config': True,
        'custom_chat': True
    }
}

llm = LocalLLM(model_name, model_cfg)
agent = AgentExecutor(llm, tool_cfg)
```



### Custom tools

We provide some default pipeline tools of multiple domain that integrates in modelscope.

Also, you can custom your tools by inheriting base tool and define names, descriptions, and parameters according to pre-defined schema. And you can implement `_local_call()` or `_remote_call()` according to your requirement. Examples of supported tool are provided below:

- Text-to-Speech Tool

```python
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

### Output Wrapper

In certain scenarios, the tool may produce multi-modal data in the form of images, audio, video, etc. However, this data cannot be directly processed by llm. To address this issue, we have implemented the `OutputWrapper` class. This class encapsulates the multi-modal data and returns a string representation that can be further processed by llm.

To use the `OutputWrapper` class, simply initialize an object with the origin multi-modal data and specify a local directory where it can be saved. The `__repr__()` function of the OutputWrapper class then returns a string that concatenates the stored path and an identifier that can be used by llm for further processing.


## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@misc{modelscope-agent,
      title={ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models},
      howpublished = {\url{https://github.com/ModelScope/modelscope-agent}},
      year={2023}
}
```
