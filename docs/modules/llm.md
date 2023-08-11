
LLM is core module of agent, which ensures the quality of interaction results.

We implement `ModelScopeGPT` and `OpenAI` for quick start. Both of them need user token to request. We have demonstrated how to use `ModelscopeGpt` in `agent.md`, and the following code provides an example of calling `OpenAI`.

```Python

model_cfg = {
    "model": "gpt-3.5-turbo-0301",
    "api_key": "your-openai-key",
    "api_base": "your-openai-base"
}

# custom template for openai
prompt_generator = MSPromptGenerator(DEFAULT_CHATGPT_PROMPT_TEMPLATE)
# openai llm
llm = OpenAi(model_cfg)

# instantiate agent with open ai llm
agent = AgentExecutor(llm, tool_cfg, additional_tool_list=additional_tool_list, prompt_generator=prompt_generator)

# define shell tool
shell_tool = LangchainTool(ShellTool())

additional_tool_list = {
    shell_tool.name: shell_tool,
}

agent.run('use tool to execute command \'ls\'')

```

![terminal file](resource/terminal-file.png)


### Local llm

You can also use open-source LLM from ModelScope or Huggingface and inference locally by `LocalLLM` class. Note that you must specify the name of the corresponding LLM so that the correct configuration can be loaded.


```Python
# local llm cfg
model_name = 'baichuan-7b'
model_cfg = {
    'baichuan-7b':{
        # model base information
        "model_id": "baichuan-inc/baichuan-7B",
        "model_revision": "v1.0.5",
        "use_lora": true,
        # lora_cfg
        "lora_cfg": {
            "replace_modules": ["pack"],
            "rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0,
            "pretrained_weights": "path/to/your/weights"
        },
        # generate_cfg
        "generate_cfg": {
            "max_new_tokens": 512,
            "do_sample": true
        }
    }
}

llm = LocalLLM(model_name, model_cfg)
agent = AgentExecutor(llm, tool_cfg)
```

Moreover, we implement `load_from_lora()` function which enables user to load your custom weights. You can invoke this function by set `use_lora=True` in config file and give corresponding lora config.

```Python
def load_from_lora(self, lora_config: LoRAConfig):

    model = self.model.bfloat16()
    # transform to lora
    Swift.prepare_model(model, lora_config)

    self.model = model
```


### Custom your LLM

If you want to use other llm, you can inherit base class and implement attributes or functions below.

- `name`: Name of llm. The related configuration is loaded through name mapping in configuration file.
- `__init__()`: This function shoule be implemented like this:
```Python
    def __init__(self, cfg):
        super().__init__(cfg)
        self.para_a = self.cfg.get('para_a', '')
        self.para_b = self.cfg.get('para_b', '')
```
- `generate()`: Directly return final response
- `stream_generate()`: Return a generator of step response, it can be used when you want to deploy your application in gradio.
