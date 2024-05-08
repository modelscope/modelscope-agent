import os
from typing import List

from assistant_api.models import (AgentConfig, ChatRequest, ChatResponse,
                                  LLMConfig, ToolResponse)
from assistant_api.server_utils import EmbeddingSingleton
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.memory import MemoryWithRetrievalKnowledge
from modelscope_agent.rag.knowledge import BaseKnowledge

DEFAULT_KNOWLEDGE_PATH = 'knowledges'
DEFAULT_INDEX_PATH = 'index'
app = FastAPI()


@app.on_event('startup')
async def startup_event():
    if not os.path.exists(DEFAULT_KNOWLEDGE_PATH):
        os.makedirs(DEFAULT_KNOWLEDGE_PATH)


@app.post('/assistant/upload_files')
async def upload_files(uuid_str: str = Form(...),
                       files: List[UploadFile] = File(...)):
    if files:
        knowledge_path = os.path.join(DEFAULT_KNOWLEDGE_PATH, uuid_str)
        if not os.path.exists(knowledge_path):
            os.makedirs(knowledge_path)
        # memory = MemoryWithRetrievalKnowledge(
        #     storage_path=knowledge_path,
        #     name=uuid_str,
        #     use_knowledge_cache=True,
        #     embedding=EmbeddingSingleton().get_embedding())
        save_dirs = []
        for file in files:
            save_dir = os.path.join(knowledge_path, file.filename)
            if os.path.exists(save_dir):
                continue
            with open(save_dir, 'wb') as f:
                f.write(file.file.read())
            # memory.run(None, url=save_dir)
            save_dirs.append(save_dir)
        print(save_dirs)
        # memory = BaseKnowledge(
        #     knowledge_source=save_dirs,
        #     cache_dir=os.path.join(knowledge_path, DEFAULT_INDEX_PATH),
        #     llm=None)
        return JSONResponse(content={
            'status': 'upload files success',
            'files': save_dirs
        })
    return JSONResponse(content={'status': 'upload fiels failed'})


@app.post('/assistant/chat')
async def chat(agent_request: ChatRequest):
    uuid_str = agent_request.uuid_str

    # agent related config
    llm_config = agent_request.llm_config.dict()
    agent_config = agent_request.agent_config.dict()
    function_list = agent_config['tools']

    # message and history
    message = agent_request.messages
    history = message[:-1]
    query = message[-1]['content']

    ref_doc = None
    if agent_request.use_knowledge:
        knowledge_path = os.path.join(DEFAULT_KNOWLEDGE_PATH, uuid_str)
        if not os.path.exists(knowledge_path):
            os.makedirs(knowledge_path)
        memory = BaseKnowledge(
            knowledge_source=[],
            cache_dir=os.path.join(knowledge_path, DEFAULT_INDEX_PATH),
            llm=llm_config)
        ref_doc = memory.run(query, files=agent_request.files)
    agent = RolePlay(
        function_list=function_list,
        llm=llm_config,
        instruction=agent_config['instruction'],
        uuid_str=uuid_str)
    result = agent.run(query, history=history, ref_doc=ref_doc)
    del agent

    if agent_request.stream:
        return StreamingResponse(result)
    else:
        response = ''
        for chunk in result:
            response += chunk
    return response


@app.post('/v1/chat/completion')
async def chat_completion(agent_request: ChatRequest):
    uuid_str = agent_request.uuid_str

    # agent related config
    llm_config = agent_request.llm_config.dict()
    agent_config = agent_request.agent_config.dict()
    function_list = agent_config.pop('tools')

    # message and history
    message = agent_request.messages
    history = message[:-1]
    query = message[-1]['content']

    ref_doc = None
    if agent_request.use_knowledge:
        knowledge_path = os.path.join(DEFAULT_KNOWLEDGE_PATH, uuid_str)
        if not os.path.exists(knowledge_path):
            os.makedirs(knowledge_path)
        memory = BaseKnowledge(
            knowledge_source=[],
            cache_dir=os.path.join(knowledge_path, DEFAULT_INDEX_PATH),
            llm=llm_config)
        ref_doc = memory.run(query, files=agent_request.files)
    agent = RolePlay(
        function_list=None,
        llm=llm_config,
        instruction=agent_config['instruction'],
        uuid_str=uuid_str)
    result = agent.run(
        query,
        history=history,
        ref_doc=ref_doc,
        tools=function_list,
        chat_mode=True)

    del agent

    # if agent_request.stream:
    #     return StreamingResponse(result)
    # else:
    llm_result = ''
    for chunk in result:
        llm_result += chunk

    response = ChatResponse(response=llm_result)

    # use re to detect tools
    try:
        import re
        import json
        result = re.search(r'Action: (.+)\nAction Input: (.+)', llm_result)
        action = result.group(1)
        action_input = json.loads(result.group(2))
        response.require_actions = True
        response.tool = ToolResponse(name=action, inputs=action_input)
    except RuntimeError:
        pass

    if agent_request.stream and response.require_actions:
        raise ValueError('Cannot stream response with tool actions')
    elif agent_request.stream:
        return StreamingResponse(response)
    else:
        return response
