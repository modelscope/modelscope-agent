# Tools Module

We offer a variety of multi-domain default pipeline tools integrated within ModelScope.

Additionally, you can customize tools by inheriting the basic ones and define the name, description, and parameters according to predefined schemas. You can implement `call()` as per your own requirements. An example of a custom tool is provided:

```python
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.tools import register_tool

# Register a custom tool name.
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

You can also refer to the demo we provide for guidance on how to use a new tool in [demo_register_new_tool](../../../examples/tools/register_new_tool.ipynb)

Furthermore, if the tool is a `langchain tool`, you can directly use our `LangchainTool` to wrap and adapt it to the current framework.

```python
from langchain.tools import ShellTool
from modelscope_agent.tools.langchain_proxy_tool import LangchainTool
from modelscope_agent.agents import RolePlay
from modelscope_agent.tools.base import TOOL_REGISTRY

# All LangchainTools should be registered in this manner, with 'terminal' being the tool name that will change according to the task.
TOOL_REGISTRY['terminal'] = LangchainTool

role_template = '你是一个助手，试图用工具帮助人类解决问题。'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

# ShellTool() is used to initialize the terminal tool.
function_list = ["code_interpreter", {'terminal':ShellTool()}]
bot = RolePlay(function_list=function_list,llm=llm_config,instruction=role_template)
# run agent
response = bot.run("请输出查看环境中git的版本号", remote=False, print_info=True)
text = ''
for chunk in response:
    text += chunk
print(text)
```

result
- Terminal

```shell
# The output from the first call to the LLM.
要查看环境中Git的版本号，我们可以使用Git自带的`--version`选项来获取该信息。接下来，我将通过终端（Terminal）执行相应的Git命令并显示版本号。

工具调用
Action: terminal
Action Input: {"commands": "git --version"}
# The output from the second call to the LLM.
Observation: <result>git version 2.37.1 (Apple Git-137.1)
</result>
Answer:当前环境中安装的Git版本号为 **2.37.1**。这是Apple提供的Git版本，具体标识为“Apple Git-137.1”。
```

## Common Tools
- `image_gen`: [Wanx Image Generation](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-wanxiang). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) Configuration is required in the environment variables.
- `code_interpreter`: [Code Interpreter](https://jupyter-client.readthedocs.io/en/5.2.2/api/client.html)
- `web_browser`: [Web Browsing](https://python.langchain.com/docs/use_cases/web_scraping)
- `amap_weather`: [AMAP Weather](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather). AMAP_TOKEN Configuration is required in the environment variables.
- `wordart_texture_generation`: [Word art texture generation](https://help.aliyun.com/zh/dashscope/developer-reference/wordart). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) Configuration is required in the environment variables.
- `web_search`: [Web Searching](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview). []
- `qwen_vl`: [Qwen-VL image recognition](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) Configuration is required in the environment variables.
- `style_repaint`: [Character style redrawn](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-wanxiang-style-repaint). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) Configuration is required in the environment variables.
- `image_enhancement`: [Chasing shadow-magnifying glass](https://github.com/dreamoving/Phantom). [DASHSCOPE_API_KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) Configuration is required in the environment variables.
- `text-address`: [Geocoding](https://www.modelscope.cn/models/iic/mgeo_geographic_elements_tagging_chinese_base/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) Configuration is required in the environment variables.
- `speech-generation`: [Speech generation](https://www.modelscope.cn/models/iic/speech_sambert-hifigan_tts_zh-cn_16k/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) Configuration is required in the environment variables.
- `video-generation`: [Video generation](https://www.modelscope.cn/models/iic/text-to-video-synthesis/summary). [MODELSCOPE_API_TOKEN](https://www.modelscope.cn/my/myaccesstoken) Configuration is required in the environment variables.
