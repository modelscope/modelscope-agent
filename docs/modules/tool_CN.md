# Tools

我们提供了一些在modelscope中集成的多域默认pipeline tools。

此外，您还可以通过继承基本工具来自定义工具，并根据预定义的模式定义名称、描述和参数。您可以根据自己的需求实现`call()`。提供了一个自定义工具的示例：

```python
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.tools import register_tool

# 注册自定义工具名称
@register_tool('RenewInstance')
class AliyunRenewInstanceTool(BaseTool):
    description = '续费一台包年包月ECS实例'
    name = 'RenewInstance'
    parameters: list = [{
        'name': 'instance_id',
        'description': 'ECS实例ID',
        'required': True,
        'type': 'string'
    }, {
        'name': 'period',
        'description': '续费时长以月为单位',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs):
        params = self._verify_args(params)
        instance_id = params['instance_id']
        period = params['period']
        return str({'result': f'已完成ECS实例ID为{instance_id}的续费，续费时长{period}月'})
```

您也可以参考我们提供的演示，了解如何在[demo_register_new_tool](../../demo/demo_register_new_tool.ipynb)中使用新工具的指南


此外，如果该工具是`langchain tool`，您可以直接使用我们的`LangchainTool`来包装和适应当前的框架。

```python
from langchain.tools import ShellTool
from modelscope_agent.tools.langchain_proxy_tool import LangchainTool
from modelscope_agent.agents import RolePlay
from modelscope_agent.tools import TOOL_REGISTRY

# All LangchainTool should be registered in such way, terminal is the tool name that will change by task
TOOL_REGISTRY['terminal'] = LangchainTool

role_template = '你是一个助手，试图用工具帮助人类解决问题。'

llm_config = {
    'model': 'qwen-max', 
    'model_server': 'dashscope',
    }

# ShellTool() use to initialize the terminal tool
function_list = ["code_interpreter", {'terminal':ShellTool()}]
bot = RolePlay(function_list=function_list,llm=llm_config,instruction=role_template)
```

## 常用工具
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