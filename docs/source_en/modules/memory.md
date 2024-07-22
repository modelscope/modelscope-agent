# Agent Memory Module

Agents utilize the memory module to facilitate multi-turn conversations, a local knowledge base, and memory persistence, achieving unified management of agent memory by constructing the memory.

## build memory

By constructing memory, the agent's memory information is managed in a unified manner.

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

- storage_path: The storage address of the memory configuration file.
- name: The name of the constructed memory.
- memory_path: The memory persistence file containing historical information, with different users having distinct IDs, stored in the corresponding ID directories.

## Implementing agent access to the local knowledge base through memory

Memory can create a Langchain VectorStore by reading local files to enable access to the local knowledge base, allowing the agent to provide targeted responses based on the knowledge base.

Download the local corpus.

```shell
wget -P /root/you_data/ https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/modelscope_qa.txt
```

The agent implements responses using the local knowledge base.


```python
# Configure environment variables; this step can be skipped if you have already set the API key in your runtime environment in advance.
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY

# Select RolePlay to configure the agent.
from modelscope_agent.agents import RolePlay

role_template = '你扮演一个python专家，需要给出解决方案'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []

bot = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# Run memory to build vector indices, read local text, and obtain results.
ref_doc = memory.run(
            query="pip install的时候有些包下载特别慢怎么办", url="/root/you_data/modelscope_qa.txt", checked=True)

# The agent performs inference with the knowledge base information returned by memory.
response = bot.run("pip install的时候有些包下载特别慢怎么办", remote=False, print_info=True, ref_doc=ref_doc)

text = ''
for chunk in response:
    text += chunk
print(text)
```

## Implement multi-turn dialogue functionality in the agent through memory.

Memory can manage the history information during the agent's execution uniformly, providing previous conversation memory before each dialogue, thereby enabling the capability for multi-turn conversations.

The agent implements multi-turn dialogue.

```python
# Configure environment variables; this step can be skipped if you have already set the API key in your runtime environment in advance.
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY

# Select RolePlay to configure the agent.
from modelscope_agent.agents import RolePlay

role_template = '你扮演一个历史人物专家，了解从古至今的历史人物'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []

bot = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# Retrieve historical information from memory.
history = memory.get_history()
# The agent performs inference based on the historical information provided by memory.
input_text = "介绍一下奥本海默"
response = bot.run("介绍一下奥本海默", remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)

# Update the historical information in memory.
from modelscope_agent.schemas import Message
if len(history) == 0:
    memory.update_history(Message(role='system', content=bot.system_prompt))
memory.update_history([
                Message(role='user', content=input_text),
                Message(role='assistant', content=text),
            ])

# Retrieve historical information again to perform inference with the agent.
history = memory.get_history()
input_text = "他是哪国人？"
response = bot.run(input_text, remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)
```

## Implement agent history persistence through memory.

Memory can persist user history information by saving historical information at a specified path, allowing users to continue the previous agent conversation when they visit again after exiting, achieving persistence of user history information.

Persist historical information to meet the invocation needs of different agents.

```python
# Configure environment variables; this step can be skipped if you have already set the API key in your runtime environment in advance.
import os
os.environ['DASHSCOPE_API_KEY']=YOUR_DASHSCOPE_API_KEY

# Select RolePlay to configure the agent.
from modelscope_agent.agents import RolePlay

role_template = '你扮演一个明星人物专家，了解从古至今的电影明星'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []

bot1 = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# Perform inference with agent A.
history = memory.get_history()
input_text = "介绍一下成龙和李连杰"
response = bot1.run(input_text, remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)

# Save memory history to a local file.
from modelscope_agent.schemas import Message
if len(history) == 0:
    memory.update_history(Message(role='system', content=bot1.system_prompt))
memory.update_history([
                Message(role='user', content=input_text),
                Message(role='assistant', content=text),
            ])
memory.save_history()

# Create agent B.
role_template = '你扮演一个电影专家，了解从古至今的电影'

llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }

function_list = []
bot2 = RolePlay(function_list=function_list, llm=llm_config, instruction=role_template)

# Read the saved history file on agent B for inference.
history = [message.model_dump() for message in memory.load_history()]
input_text = "他俩共同出演了哪一步电影？"
response = bot2.run(input_text, remote=False, print_info=True, history=history)
text = ''
for chunk in response:
    text += chunk
print(text)
```
