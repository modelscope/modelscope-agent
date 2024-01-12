from modelscope_agent.agents import RolePlay

# config
role_template = '你扮演一个天气预报助手，你需要查询相应地区的天气，并调用给你的画图工具绘制一张城市的图。'
llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
function_list = ['amap_weather', 'image_gen']

# init agent
bot = RolePlay(
    function_list=function_list, llm=llm_config, instruction=role_template)

# run agent
response = bot.run('朝阳区天气怎样？')

# result processing
text = ''
for chunk in response:
    text += chunk
print(text)
