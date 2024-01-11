from modelscope_agent.agents.function_calling import FunctionCalling

llm_config = {'model': 'qwen-max', 'model_server': 'openai'}

llm_config['api_base'] = input()

# input tool name
# function_list = ['amap_weather']

# input tool args
function_list = [{'name': 'amap_weather'}]

bot = FunctionCalling(function_list=function_list, llm=llm_config)

response = bot.run('朝阳区天气怎样？')

text = ''
for chunk in response:
    text += chunk
print(text)
