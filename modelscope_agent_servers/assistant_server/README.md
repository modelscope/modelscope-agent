## launch service

To launch assistant service, you can use `uvicorn` as the server. Suppose you are in directory of `modelscope-agent`, the command to launch service is:

```
uvicorn modelscope_agent_servers.assistant_server.api:app --reload
```

## use case

Currently, we provide three apis. These apis need to associate with `uuid`.

* `assistant/upload_files`: upload files for knowledge retrieval.
* `assistant/chat`: start a conversation use agents. This call will complete a single round with agents, including **tool execution**.
* `v1/chat/completion`: start a conversation like openai chat completion, support function calling and knowledges retrieval.

### upload files

You can upload your local files use `requests` library. Below is a simple example:

```Python
import requests
# you may need to replace with your sever url
url1 = 'http://localhost:8000/assistant/upload_files'

# uuid
datas = {
    'uuid_str': 'test'
}

# the files to upload
files = [
    ('files', ('ms.txt', open('ms.txt', 'rb'), 'text/plain'))
]
response = requests.post(url1, data=datas, files=files)

```

### v1/chat/completion

To interact with the chat API, you should construct a object like `ChatRequest` on the client side, and then use the requests library to send it as the request body.

#### function calling
An example code snippet is as follows:

```Python
url = 'http://localhost:8000/v1/chat/completion'

# 要发送的数据
llm_cfg = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.environ.get('DASHSCOPE_API_KEY'),
}

tool_cfg = [{
    'name': 'amap_weather',
    'description': 'amap weather tool',
    'parameters': [{
        'name': 'location',
        'type': 'string',
        'description': '城市/区具体名称，如`北京市海淀区`请描述为`海淀区`',
        'required': True
    }]
}]
agent_cfg = {
    'tools': tool_cfg,
    'name': 'test',
    'description': 'test assistant',
    'instruction': 'you are a helpful assistant'
}


request = {
    'agent_config': agent_cfg,
    'llm_config': llm_cfg,
    'messages': [
        {'content': '朝阳区天气', 'role': 'user'}
    ],
    'uuid_str': 'test',
    'stream': False
}

response = requests.post(url, json=request)

# 输出响应内容
print(response.text)

```

With above examples, the output may like this:
```Python
{"request_id":"xxxxx",
"message":"",
"output":{
    "response":"Action: amap_weather\nAction Input: {\"location\": \"朝阳区\"}\n",
    "require_actions":true,
    "tool":{"name":"amap_weather","inputs":{"location":"朝阳区"}}}}
```

#### knowledge retrieval

To enable knowledge retrieval, you'll need to include use_knowledge and files in your configuration settings.

- `use_knowledge`: Specifies whether knowledge retrieval should be activated.
- `files`: the file(s) you wish to use during the conversation. By default, all previously uploaded files will be used.

```Python
request = {
    'agent_config': agent_cfg,
    'llm_config': llm_cfg,
    'messages': [
        {'content': '高德天气API申请', 'role': 'user'}
    ],
    'uuid_str': 'test',
    'stream': False,
    'use_knowledge': True, # use knowledge retrieval
    'files': ['QA.pdf'] # the files you want to use in this conversation
}
datas = {
    'agent_request': json.dumps(request)
}
response = requests.post(url, json=request)
```

### assistant/chat

Like `v1/chat/completion` API, you should construct a `ChatRequest` object when use `assistant/chat`.


```Python
url = 'http://localhost:8000/assistant/chat'

# llm config
llm_cfg = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.environ.get('DASHSCOPE_API_KEY'),
}
# agent config
agent_cfg = {
    'name': 'test',
    'description': 'test assistant',
    'instruction': 'you are a helpful assistant'
}

#
request = {
    'agent_config': agent_cfg,
    'llm_config': llm_cfg,
    'messages': [
        {'content': '请为我介绍一下modelscope', 'role': 'user'}
    ],
    'uuid_str': 'test',
    'use_knowledge': True # whether to use knowledge
    'files': ['ms.txt'] # you can specify the file you want to use in this conversation.
}

response = requests.post(url, json=request)

```

If you want to use `stream` output, you can extract messages like this:

```Python
request = {
    'agent_config': agent_cfg,
    'llm_config': llm_cfg,
    'messages': [
        {'content': '请为我介绍一下modelscope', 'role': 'user'}
    ],
    'uuid_str': 'test',
    'stream': True, # whether to use stream
    'use_knowledge': True
}

response = requests.post(url, json=request)

# extract message
if response.encoding is None:
    response.encoding = 'utf-8'

for line in response.iter_lines(decode_unicode=True):
    if line:
        print(line)
```
