# LLM Module
LLM is core module of agent, which ensures the quality of interaction results.

We implement `dashscope` and `zhipu` for quick start. Both of them need user token to request.
- `dashscope`: To use the model provided by dashscope, you need to configure DASHSCOPE_API_KEY in the environment variable. You can get your dashscope api key at [here](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key).
- `zhipu`: To use the model provided by ZhipuAI, you need to configure ZHIPU_API_KEY in the environment variable. You can get your dashscope api key at [here](https://open.bigmodel.cn/usercenter/apikeys).


You can also use open-source LLM from ModelScope or Huggingface and inference locally. You need to first pull up the openai format service API through vllm or other methods, and configure model_server to `openai` when using it. Note that you must specify the name of the corresponding LLM so that the correct configuration can be loaded.

## How to use
In general, you do not need to initialize llm directly. In the initialization method provided by the base class `agent`, the get_chat_model method will be called. Just pass in llm_config, and llm will automatically initialize it for you in `agent`.

### get_chat_model
```Python
from modelscope_agent.llm import get_chat_model
llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

llm = get_chat_model(**llm_config)
```

It will automatically find the appropriate one from the registered llm classes for initialization according to the llm_config configuration.

- how to register
```python
# modelscope_agent/llm/dashscope.py
from .base import BaseChatModel, register_llm

@register_llm('dashscope')
class DashScopeLLM(BaseChatModel):
    pass

@register_llm('dashscope_qwen')
class QwenChatAtDS(DashScopeLLM):
    pass

# modelscope_agent/llm/__init__.py
from .dashscope import DashScopeLLM, QwenChatAtDS
__all__ = [..., ..., 'DashScopeLLM', 'QwenChatAtDS',]
```

The selection of this class follows the following rules:
- First search for "`model_server`_{prefix of `model`}" from the registered class
For example, `{'model': 'qwen-max', 'model_server': 'dashscope'}`, first match `dashscope_qwen`
- If not found, matches "`model_server`"
- If none of the above names have been registered, an error will be reported

### Specify class
You can also initialize the specified class and then pass in the agent.
```Python
from modelscope_agent.llm import DashScopeLLM
llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

llm = DashScopeLLM(**llm_config)
```

### Customize your LLM

If you want to use other llm, you can inherit base class and implement attributes or functions below.

- `_chat_stream`: 子类一定要实现的方法，对应流式输出的chat方法，接收messages格式作为输入。
- `_chat_no_stream`: 子类一定要实现的方法，对应非流式输出的chat方法，接收messages格式作为输入。

```Python
from .base import BaseChatModel, register_llm

@register_llm('your_custom_model')
class YourCustomLLM(BaseChatModel):
    def _chat_no_stream(self, messages: List[Dict], **kwargs):
        output_text = self.model.chat(messages, **kwargs)
        return str(output_text)

    def _chat_stream(self, messages: List[Dict], **kwargs):
        output_text = self.model(messages, **kwargs)
        yield str(output_text)
```
其中输入参数messages为openai格式
```python
messages = [{
    'role': 'user',
    'content': 'Hello.'
}, {
    'role': 'assistant',
    'content': 'Hi there!'
}, {
    'role': 'user',
    'content': 'Tell me a joke.'
}]
```
