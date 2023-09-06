# Introduction

**ModelScope-Agent**, a general and customizable agent framework for real-world applications, based on open-source LLMs as controllers. It provides a user-friendly system library that are:
- **cutomizable engine**: customizable engine design to support model training on multiple open-source LLMs
- **Diversified and Comprehensive APIs**: enabling seamless integration with both model APIs and common APIs in a unified way.

To equip the LLMs with tool-use abilities, a comprehensive framework has been proposed spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications.


## How to start

The agent incorporates an LLM along with task-specific tools, and uses the LLM to determine which tool or tools to invoke in order to complete the user's tasks.

To start, all you need to do is initialize an `LLM` object and an `AgentExecutor` object with corresponding tasks

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

We provide several default tools that have been adapted from the **ModelScope** model pipeline, and these tools will be automatically initialized. These tools can be invoked via the local inference pipeline, or by a remote deployed server by setting `remote=True`.

* `Text address`:
* `Text ie`:
* `Text ner`:
* `Text translation(Chinese to English)`:
* `Text translation(English to Chinese)`:
* `Text to image`:
* `Text to speech`:
* `Text to video`:
* `Image chat`:

Here are some examples of task executions:

```Python

# write slogan and transfer to speech in remote mode
agent.run("写一个 2023上海世界人工智能大会 20 字以内的口号，并语音念出来", remote=False)

# extract address in sentence in local mode
agent.run('使用地址识别模型，从下面的地址中找到省市区等元素，地址：浙江杭州市江干区九堡镇三村村一区', remote=True)

```


## Custom agents

An `AgentExecutor` object consists of the following components:

- `LLM`: A large language model that is responsibile to process your inputs and decide calling tools.
- `tool_list`: A list consists of available tools for agents.
- `PromptGenerator`: A module integrates `prompt_template`, `user_input`, `history`, `tool_list`... into final prompt for llm.
- `OutputParser`: A module to parse llm response into the tools to be invoked and the corresponding parameters.

We provide default implement of these components for users, but you can also custom your components according to your requirement.

### `PromptGenerator`

To custom you `PromptGenerator`, you may need to override the following functions in base class.

```Python
class MyPromptGenerator(PromptGenerato):

    def init_prompt(self, task, tool_list, available_tool_list):
        """
        in this function, specify how to initialize your prompt.
        """
        ...
        return task

    def generate(self, llm_result, exec_result):
        """
        the agent may need to interact with llm multiple times. This function generate next round prompt based on previous llm_result and exec_result and update history
        """
        ...
        self.prompt += f'{llm_result}{exec_result}'

        return self.prompt
```

### `OutputParser`

```Python
class MyOutputParser(OutputParser):

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """
        in this function, you need to define how to parse and get action(str) and action paramerers(dict)
        """
        return resonse, {}

```

### `Tool`

To custom your tool, please refer to `tool.md`

To use custom tools with the agent, you should specify them using the `additional_tool_list` parameter during agent initialization.

Additionally, the `tool_list` of agent may contain tools that are not relevant to your task. You can specify the tools that are available to the agent for a particular task by using the `set_available_tools` function.

```Python

# define your tool
my_tool = MyTool()
...

additional_tool_list = {
    'my_tool': my_tool
}

# initialize with additional_tool_list
agent = AgentExecutor(llm, tool_cfg, additional_tool_list=additional_tool_list)

available_tool_list = [
    'my_tool'
]

# set available_tool list
agent.set_available_tools(available_tool_list)

```


### `LLM`

To custom your llm, please refer to `llm.md`

### Configuration`

For configurations like `user_token` that are not meant to be public, we recommend using the `dotenv` package along with an `.env` file to store these settings securely.

Specifically, we provide an `.env.template` file and corresponding config files in our repository. You can easily customize the configuration by referring to the provided example and utilize your own `.env` file to read the configuration settings.
