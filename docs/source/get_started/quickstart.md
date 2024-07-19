# 快速开始

agent结合了大型语言模型（LLM）以及特定任务的工具，并利用LLM来确定为了完成用户任务需要调用哪个或哪些工具。

在一开始，您所需要做的就是使用相应的任务初始化一个`RolePlay`对象。

- 样本代码使用了 qwen-max 模型、绘图工具和天气预报工具。
  - 使用 qwen-max 模型需要将示例中的 YOUR_DASHSCOPE_API_KEY 替换为您的 API-KEY，以便代码正常运行。您的 YOUR_DASHSCOPE_API_KEY 可以在[这里](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)获得。绘图工具也调用了 DASHSCOPE API（wanx），因此不需要额外配置。
  - 在使用天气预报工具时，需要将示例中的 YOUR_AMAP_TOKEN 替换为您的高德天气 API-KEY，以便代码能够正常运行。您的 YOUR_AMAP_TOKEN 可以在[这里](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather)获得。

```Python
# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY
os.environ['AMAP_TOKEN']=YOUR_AMAP_TOKEN

# 选用RolePlay 配置agent
from modelscope_agent.agents.role_play import RolePlay  # NOQA

role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

# input tool name
function_list = ['amap_weather', 'image_gen']

bot = RolePlay(
    function_list=function_list, llm=llm_config, instruction=role_template)

response = bot.run('朝阳区天气怎样？')

text = ''
for chunk in response:
    text += chunk
```

结果
- Terminal 运行

```shell
# 第一次调用llm的输出
Action: amap_weather
Action Input: {"location": "朝阳区"}

# 第二次调用llm的输出
目前，朝阳区的天气状况为阴天，气温为1度。

Action: image_gen
Action Input: {"text": "朝阳区城市风光", "resolution": "1024*1024"}

# 第三次调用llm的输出
目前，朝阳区的天气状况为阴天，气温为1度。同时，我已为你生成了一张朝阳区的城市风光图，如下所示：

![](https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1d/45/20240204/3ab595ad/96d55ca6-6550-4514-9013-afe0f917c7ac-1.jpg?Expires=1707123521&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=RsJRt7zsv2y4kg7D9QtQHuVkXZY%3D)
```
