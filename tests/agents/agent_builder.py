from modelscope_agent.agents.agent_builder import AgentBuilder

llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

# input tool name
# function_list = ['amap_weather']

# input tool args
function_list = [{'name': 'amap_weather'}]

bot = AgentBuilder(function_list=function_list, llm=llm_config)

response = bot.run('创建一个多啦A梦')

text = ''
for chunk in response:
    text += chunk
print(text)
