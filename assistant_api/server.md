## launch service

To launch assistant service, you can use `uvicorn` as the server. Suppose you are in directory of `modelscope-agent`, the command to launch service is:

```
uvicorn assistant_api.server:app --reload
```

## use case

Currently, we provide two apis. These two apis may need to associate with `uuid`.

* `chat`: start a conversation use agents.
* `upload_files`: upload files for knowledge retrieval.

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

### chat

To interact with the chat API, you should construct a object like `ChatRequest` on the client side, and then use the requests library to send it as the request body. An example code snippet is as follows:


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
