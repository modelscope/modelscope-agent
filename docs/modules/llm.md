
LLM is core module of agent, which ensures the quality of interaction results.

The initialization of LLM is primarily achieved through `LLMFactory.build_llm`. You need to provide the LLM name and the corresponding configuration.

We implement `ModelScopeGPT` and `OpenAI` for quick start. Both of them need user token to request.

You can also use open-source LLM from ModelScope or Huggingface and inference locally by `LocalLLM` class. Note that you must specify the name of the corresponding LLM so that the correct configuration can be loaded.


An example for importing a local LLM with `LLMFactory.build_llm` is shown below.


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


llm = LLMFactory.build_llm(model_name, model_cfg)
```

Currently, configuration of `LocalLLM` may contains following parameters:

- `model_cls`: model class for load LLM, should be corresponding with `model_id`. Default `AutoModelForCausalLM`.
- `tokenizer_cls`: tokenizer class for tokenizer, should be corresponding with `model_id`. Default `AutoTokenizer`.
- `generation_cfg`: Config of response generations.
- `use_raw_generation_config`: Whether to use raw generation config defined in modelscope hub.
- `use_lora`: Whether to load custom LoRA weight, should be used with `lora_ckpt_dir`.
- `lora_ckpt_dir`: LoRA checkpoint directory.
- `custom_chat`: Whether to use build-in `chat()` function of model.
- `end_token`: Words to truncate the response since some LLM may generate repetitive response.
- `include_end`: Whether to include end_token in final response.

Moreover, we implement `load_from_lora()` function with modelscope swift library, which enables user to load their custom lora weight. You can invoke this function by set `use_lora=True` in config file and give corresponding lora config.

```Python
def load_from_lora(self):

    model = self.model.bfloat16()
    # transform to lora
    model = Swift.from_pretrained(model, self.lora_ckpt_dir)

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
