import os
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.rag.knowledge import BaseKnowledge
from modelscope_agent_servers.assistant_server.models import (ChatRequest,
                                                              ChatResponse,
                                                              ToolResponse)
from modelscope_agent_servers.assistant_server.utils import \
    tool_calling_wrapper
from modelscope_agent_servers.service_utils import (create_error_msg,
                                                    create_success_msg)

DEFAULT_KNOWLEDGE_PATH = 'knowledges'
DEFAULT_INDEX_PATH = 'index'
app = FastAPI()


@app.on_event('startup')
async def startup_event():
    if not os.path.exists(DEFAULT_KNOWLEDGE_PATH):
        os.makedirs(DEFAULT_KNOWLEDGE_PATH)


@app.post('/v1/files')
async def upload_files(uuid_str: str = Form(...),
                       files: List[UploadFile] = File(...)):
    """

    Args:
        uuid_str: user id
        files: file

    Returns:

    """
    request_id = str(uuid4())
    if files:
        knowledge_path = os.path.join(DEFAULT_KNOWLEDGE_PATH, uuid_str)
        if not os.path.exists(knowledge_path):
            os.makedirs(knowledge_path)

        save_dirs = []
        for file in files:
            # 如果文件名是一个路径，那么就不保存
            if os.path.dirname(file.filename):
                return create_error_msg({'status': 'Invalid file name'},
                                        request_id=request_id)
            save_dir = os.path.join(knowledge_path, file.filename)
            if os.path.exists(save_dir):
                continue
            with open(save_dir, 'wb') as f:
                f.write(file.file.read())
            save_dirs.append(save_dir)
        print(save_dirs)
        _ = BaseKnowledge(
            knowledge_source=save_dirs,
            cache_dir=os.path.join(knowledge_path, DEFAULT_INDEX_PATH),
            llm=None)
        return create_success_msg(
            {
                'status': 'upload files success',
                'files': save_dirs
            },
            request_id=request_id)
    return create_success_msg({'status': 'No valid files'},
                              request_id=request_id)


@app.post('/v1/assistant/lite')
async def chat(agent_request: ChatRequest):
    uuid_str = agent_request.uuid_str
    request_id = str(uuid4())

    # agent related config
    llm_config = agent_request.llm_config.dict()
    agent_config = agent_request.agent_config.dict()
    function_list = agent_config['tools']
    use_tool_api = agent_request.use_tool_api

    # message and history
    message = agent_request.messages
    history = message[:-1]
    query = message[-1]['content']

    # additional kwargs
    kwargs = agent_request.kwargs

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
        if ref_doc == 'Empty Response':
            return create_error_msg(
                'No valid knowledge contents.', request_id=request_id)
    agent = RolePlay(
        function_list=function_list,
        llm=llm_config,
        instruction=agent_config['instruction'],
        uuid_str=uuid_str,
        use_tool_api=use_tool_api)
    result = agent.run(query, history=history, ref_doc=ref_doc, **kwargs)
    del agent

    if agent_request.stream:
        return StreamingResponse(result)
    else:
        response = ''
        for chunk in result:
            response += chunk
    return create_success_msg({'response': response}, request_id=request_id)


@app.post('/v1/chat/completion')
async def chat_completion(agent_request: ChatRequest):
    uuid_str = agent_request.uuid_str
    request_id = str(uuid4())

    # config
    llm_config = agent_request.llm_config.dict()
    function_list = agent_request.tools
    tool_choice = agent_request.tool_choice

    # message and history
    message = agent_request.messages
    history = message[:-1]
    query = message[-1]['content']

    # additional kwargs
    kwargs = agent_request.kwargs

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
        if ref_doc == 'Empty Response':
            return create_error_msg(
                'No valid knowledge contents.', request_id=request_id)
    agent = RolePlay(function_list=None, llm=llm_config, uuid_str=uuid_str)
    result = agent.run(
        query,
        history=history,
        ref_doc=ref_doc,
        tools=function_list,
        tool_choice=tool_choice,
        chat_mode=True,
        **kwargs)

    del agent

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
    except Exception:
        pass

    if agent_request.stream and response.require_actions:
        return create_error_msg(
            'not support stream with tool', request_id=request_id)
    elif agent_request.stream:
        return StreamingResponse(response)
    else:
        kwargs = {'choices': tool_calling_wrapper(response)}
        return create_success_msg(None, request_id=request_id, **kwargs)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='127.0.0.1', port=31512)
