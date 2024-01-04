from modelscope_agent.prompts.role_play import RolePlay  # NOQA

role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'

llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

# input tool name
# function_list = ['amap_weather']

# input tool args
function_list = [{'name': 'amap_weather'}]

bot = RolePlay(
    function_list=function_list, llm=llm_config, instruction=role_template)

response = bot.run('朝阳区天气怎样？')

text = ''
for chunk in response:
    text += chunk
print(text)
