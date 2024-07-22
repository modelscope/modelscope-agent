import os
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import StreamingResponse
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.llm.utils.function_call_with_raw_prompt import \
    detect_multi_tool
from modelscope_agent.rag.knowledge import BaseKnowledge
from modelscope_agent_servers.assistant_server.models import (
    AgentRequest, ChatCompletionRequest, ChatCompletionResponse, ToolResponse)
from modelscope_agent_servers.assistant_server.utils import (
    choice_wrapper, parse_messages, stream_choice_wrapper)
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


@app.post('/v1/assistants/lite')
async def chat(agent_request: AgentRequest):
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


@app.post('/v1/chat/completions')
async def chat_completion(chat_request: ChatCompletionRequest,
                          authorization: str = Header(None)):

    request_id = f'chatcmpl_{str(uuid4())}'
    user = chat_request.user
    model = chat_request.model
    # remove the prefix 'Bearer ' from the authorization header
    auth = authorization[7:] if authorization else 'EMPTY'

    # llm_config
    llm_config = {
        'model': model,
        'model_server': os.environ.get('MODEL_SERVER', 'dashscope'),
        'api_key': auth
    }

    # tool related config
    tools = chat_request.tools
    tool_choice = None
    parallel_tool_calls = True
    if tools:
        tool_choice = chat_request.tool_choice
        parallel_tool_calls = chat_request.parallel_tool_calls

    # parse meesage
    query, history, image_url = parse_messages(chat_request.messages)

    # additional kwargs

    agent = RolePlay(function_list=None, llm=llm_config, uuid_str=user)
    result = agent.run(
        query,
        history=history,
        tools=tools,
        tool_choice=tool_choice,
        chat_mode=True,
        parallel_tool_calls=parallel_tool_calls,
        # **kwargs)
    )

    if chat_request.stream:
        stream_chat_response = stream_choice_wrapper(result, model, request_id,
                                                     agent.llm)
        return StreamingResponse(
            stream_chat_response, media_type='text/event-stream')

    llm_result = ''
    for chunk in result:
        llm_result += chunk

    usage = agent.llm.get_usage()
    print(usage)

    del agent

    has_action, tool_list, _ = detect_multi_tool(llm_result)
    choices = choice_wrapper(llm_result, tool_list)

    chat_response = ChatCompletionResponse(
        choices=choices,
        model=model,
        id=request_id,
        system_fingerprint=request_id,
        usage=usage)

    return create_success_msg(
        None, request_id=request_id, **chat_response.dict())


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='127.0.0.1', port=31512)
