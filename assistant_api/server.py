import os
from typing import List

from assistant_api.models import AgentConfig, ChatRequest, LLMConfig
from assistant_api.server_utils import EmbeddingSingleton
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.memory import MemoryWithRetrievalKnowledge

DEFAULT_KNOWLEDGE_PATH = 'knowledges'

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
        memory = MemoryWithRetrievalKnowledge(
            storage_path=knowledge_path,
            name=uuid_str,
            use_knowledge_cache=True,
            embedding=EmbeddingSingleton().get_embedding())
        for file in files:
            save_dir = os.path.join(knowledge_path, file.filename)
            if os.path.exists(save_dir):
                continue
            with open(save_dir, 'wb') as f:
                f.write(file.file.read())
            memory.run(None, url=save_dir)
        return JSONResponse(content={'status': 'upload files success'})
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
        memory = MemoryWithRetrievalKnowledge(
            storage_path=knowledge_path,
            name=uuid_str,
            use_knowledge_cache=True,
            embedding=EmbeddingSingleton().get_embedding())
        ref_doc = memory.run(query)

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
