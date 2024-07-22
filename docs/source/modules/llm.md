# LLM模块使用说明
## LLM
LLM是agent的核心模块，它确保了交互结果的质量

我们实现了`dashscope`和`zhipu`以方便快速上手，两者都需要用户token来发起请求

- `dashscope`: 要使用dashscope提供的模型，您需要在环境变量中配置DASHSCOPE_API_KEY。您可以在[此处](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)获取您的dashscope api key。
- `zhipu`: 要使用ZhipuAI提供的模型，您需要在环境变量中配置ZHIPU_API_KEY。您可以在[此处](https://open.bigmodel.cn/usercenter/apikeys)获取您的Zhipu API密钥。


您也可以使用ModelScope或Huggingface的开源LLM并在本地进行推理，您需要首先通过vllm或其他方式拉起openai格式的服务API，并在使用时将model_server配置为`openai`，请注意，您必须指定相应LLM的名称，以便加载正确的配置。

## 怎样使用

通常情况下，您不需要直接初始化llm。在基类`agent`提供的初始化方法中，会调用get_chat_model方法。只需传入llm_config，llm就会在`agent`中自动为您初始化。

### 获取chat model
```Python
from modelscope_agent.llm import get_chat_model
llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

llm = get_chat_model(**llm_config)
```

它将根据llm_config配置从已注册的llm类中自动找到合适的一个进行初始化

- 如何注册
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

此类的选择遵循以下规则：
- 首先从注册类中搜索"`model_server`_{`model`前缀}",例如,`{'model': 'qwen-max', 'model_server': 'dashscope'}`,首先匹配`dashscope_qwen`。
- 如果未找到，则匹配"model_server"
- 如果以上名称均未被注册，则会报错

### 指定类
您也可以先初始化指定的类，然后将其传入agent。

```Python
from modelscope_agent.llm import DashScopeLLM
llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

llm = DashScopeLLM(**llm_config)
```

### 自定义LLM

如果您想使用其他llm，可以继承基类并实现以下属性或函数。

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
