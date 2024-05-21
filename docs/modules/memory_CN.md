# agent memory

agent通过使用memory模块来实现多轮对话，本地知识库，以及记忆持久化，通过构建memory来实现对agent记忆的统一管理

## 构建memory
 通过构建memory来统一管理agent的记忆信息

```python
from modelscope_agent.memory import MemoryWithRetrievalKnowledge

DEFAULT_UUID_HISTORY = "/root/you_data/you_config/history"
storage_path = "/root/you_data/you_config/config"
memory_history_path = os.path.join(DEFAULT_UUID_HISTORY, 'default_user.json')
memory_agent_name = 'default_memory'
memory = MemoryWithRetrievalKnowledge(storage_path=storage_path,
        name=memory_agent_name,
        memory_path=memory_history_path,
        )
```

- storage_path: memory配置文件的存储地址
- name: 构建的memory名称
- memory_path: memory持久化保存的历史信息文件，不同用户拥有不同的id，存储在对应的id目录下

## 通过memory实现agent调用本地知识库

memory可以通过读取本地文件来创建Langchain VectorStore来实现调用本地知识库，满足agent根据知识库做出针对性的回答。

下载本地语料库
```shell
wget -P /root/you_data/ https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/modelscope_qa.txt
```

agent实现调用本地知识库进行回答

```python
# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY

# 选用RolePlay 配置agent
from modelscope_agent.agents import RolePlay

role_template = '你扮演一个python专家，需要给出解决方案'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []

bot = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# 运行memory构建向量索引，读取本地文本，获取结果
ref_doc = memory.run(
            query="pip install的时候有些包下载特别慢怎么办", url="/root/you_data/modelscope_qa.txt", checked=True)

# agent得到memory返回的知识库信息进行推理
response = bot.run("pip install的时候有些包下载特别慢怎么办", remote=False, print_info=True, ref_doc=ref_doc)

text = ''
for chunk in response:
    text += chunk
print(text)
```

## 通过memory实现agent多轮对话功能

memory可以统一管理agent执行中的history信息，在每次对话前提供先前对话记忆，从而实现多轮对话的能力

agent实现多轮对话
```python
# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY

# 选用RolePlay 配置agent
from modelscope_agent.agents import RolePlay

role_template = '你扮演一个历史人物专家，了解从古至今的历史人物'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []

bot = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# 从memory中获取历史信息
history = memory.get_history()
# agent得到memory提供的历史信息进行推理
input_text = "介绍一下奥本海默"
response = bot.run("介绍一下奥本海默", remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)

# 更新memory中的历史信息
from modelscope_agent.schemas import Message
if len(history) == 0:
    memory.update_history(Message(role='system', content=bot.system_prompt))
memory.update_history([
                Message(role='user', content=input_text),
                Message(role='assistant', content=text),
            ])

# 再次获取历史信息，使用agent进行推理
history = memory.get_history()
input_text = "他是哪国人？"
response = bot.run(input_text, remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)
```

## 通过memory实现agent history持久化
memory可以通过将历史信息保存在指定路径下，允许用户退出后，下次访问时依然可以继续上次的agent对话，做到用户历史信息的持久化。

将历史信息持久化来满足不同agent的调用

```python
# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY

# 选用RolePlay 配置agent A
from modelscope_agent.agents import RolePlay

role_template = '你扮演一个明星人物专家，了解从古至今的电影明星'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []

bot1 = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# 执行agent A的推理
history = memory.get_history()
input_text = "介绍一下成龙和李连杰"
response = bot1.run(input_text, remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)

# 将memory history保存到本地文件
from modelscope_agent.schemas import Message
if len(history) == 0:
    memory.update_history(Message(role='system', content=bot1.system_prompt))
memory.update_history([
                Message(role='user', content=input_text),
                Message(role='assistant', content=text),
            ])
memory.save_history()

# 创建agent B
role_template = '你扮演一个电影专家，了解从古至今的电影'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []
bot2 = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# 在agent B上读取保存的history文件进行推理
history = [message.model_dump() for message in memory.load_history()]
input_text = "他俩共同出演了哪一步电影？"
response = bot2.run(input_text, remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)
```
