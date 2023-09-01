# ModelScope Agent

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

agent.run('写一个 2023 上海世界人工智能大会 20 字以内的口号，并念出来')
```

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
model_name = 'baichuan-7b'
model_cfg = {
    'baichuan-7b':{
        'model_id': 'baichuan-inc/baichuan-7B',
        'model_revision': 'v1.0.5',
        'generate_cfg': {
            'max_new_tokens': 512,
            'do_sample': True
        }
    }
}

llm = LocalLLM(model_name, model_cfg)
agent = AgentExecutor(llm, tool_cfg)
```



### Custom tools

We provide some default pipeline tools of multiple domain that integrates in modelscope.

Also, you can custom your tools by inheriting base tool and define names, descriptions, and parameters according to pre-defined schema. And you can implement `_local_call()` or `_remote_call()` according to your requirement. An example of custom tool is provided below:

```python
class CustomTool(Tool):
    description = 'my custonm translation tool'
    name = 'modelscope_my-custom-translation-tool'
    parameters: list = [{
        'name': 'input',
        'description': '需要翻译的文本',
        'required': True
    }]

    def _local_call():
        ...

    def _remote_call():
        ...
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
