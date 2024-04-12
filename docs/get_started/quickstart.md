# Quickstart
The agent combines a large language model (LLM) with tools specific to certain tasks and utilizes the LLM to determine which tool or tools need to be invoked to complete a user's task.

Initially, all you need to do is initialize a `RolePlay` object with the corresponding task.

- The sample code uses the qwen-max model, a drawing tool, and a weather forecast tool.
    - To use the qwen-max model, you need to replace YOUR_DASHSCOPE_API_KEY in the example with your API-KEY for the code to function properly. You can obtain your YOUR_DASHSCOPE_API_KEY [here](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key). The drawing tool also calls the DASHSCOPE API (wanx), so no additional configuration is required.
    - When using the weather forecast tool, you need to replace YOUR_AMAP_TOKEN in the example with your Amap Weather API-KEY to ensure the code works correctly. You can obtain your YOUR_AMAP_TOKEN [here](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather).

```Python
# Configure environment variables; this step can be skipped if you have already set the API key in your runtime environment in advance.
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY
os.environ['AMAP_TOKEN']=YOUR_AMAP_TOKEN

# Select RolePlay to configure the agent.
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

result
- Terminal

```shell
# The output from the first call to the LLM.
Action: amap_weather
Action Input: {"location": "朝阳区"}

# The output from the second call to the LLM.
目前，朝阳区的天气状况为阴天，气温为1度。

Action: image_gen
Action Input: {"text": "朝阳区城市风光", "resolution": "1024*1024"}

# The output from the third call to the LLM.
目前，朝阳区的天气状况为阴天，气温为1度。同时，我已为你生成了一张朝阳区的城市风光图，如下所示：

![](https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1d/45/20240204/3ab595ad/96d55ca6-6550-4514-9013-afe0f917c7ac-1.jpg?Expires=1707123521&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=RsJRt7zsv2y4kg7D9QtQHuVkXZY%3D)
```
