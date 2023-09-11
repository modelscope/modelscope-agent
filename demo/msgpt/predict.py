from __future__ import annotations

import json
import os
import traceback
import re
from copy import deepcopy
import gradio as gr
import uuid
def stream_predict(chatbot, # ChatBot
                   user_input, # Textbox
                   upload_image_url, # imageUrl
                   agent # agent
                ):
    
    role_name = 'ModelScopeGPT'
    role_type = '自定义角色'
    bot_profile = '你是达摩院的ModelScopeGPT（魔搭助手），你是个大语言模型， 是2023年达摩院的工程师训练得到的。你有多种能力，可以通过插件集成魔搭社区的模型api来回复用户的问题，还能解答用户使用模型遇到的问题和模型知识相关问答。'
    verbose_response = 'yes'
    role_instruction = ''
    search_instruction = ''
    search_keyword_expand = 'no'
    force_search = ''
    verbose_full_prompt = 'yes'

    if not user_input:
        if len(history) == 0:
            yield chatbot, "请输入问题……"
            return
        else:
            user_input = chatbot[-1][0]
            chatbot = chatbot[:-1]
            yield chatbot, "重新生成回答……"
    else:
        if upload_image_url:
            user_input = f"![upload]({upload_image_url})\n" + user_input
        print("user_input:", user_input)
        yield chatbot, "开始生成回答……"

    chatbot.append((user_input, "处理中..."))

    yield chatbot, "开始实时传输回答……"

    response = ''
    try:
        for frame in agent.stream_run(user_input, remote=False):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '')
            error_message = frame.get("error")
            if is_final:
                chatbot[-1] = (chatbot[-1][0], response)
                yield chatbot, ''
                break
            elif error_message:
                chatbot[-1] = (chatbot[-1][0], error_message)
                yield chatbot, ''
            else:
                if len(exec_result) != 0:
                    # llm_result
                    frame_text = f'\n\n<|startofexec|>{exec_result}<|endofexec|>\n'
                else:
                    # action_exec_result
                    frame_text = llm_result
                response = f'{response}{frame_text}'
                chatbot[-1] = (chatbot[-1][0], response)
                yield chatbot, ''
    except:
        traceback.print_exc()
        chatbot[-1] = (chatbot[-1][0], 'chat_async error.')
        yield chatbot,  ''


def upload_image(
    file
):
    # file_ext = os.path.splitext(file.name)[-1]
    # base_dir = 'oss://xdp-expriment/modelscope/image/'
    # event_id = uuid.uuid4().hex[:16]
    # oss_url = os.path.join(base_dir, f"user-upload-{event_id}{file_ext}")
    # print('file:', file.name, file_ext, oss_url)
    # io.copy(file.name, oss_url)
    # io.authorize(oss_url)
    # bucket_name, path = io._split_name(oss_url)
    # generated_oss_url = io.buckets[bucket_name]._make_url(bucket_name, path)
    # generated_oss_url = generated_oss_url.replace('http:', 'https:')
    # print('ossUrl:', generated_oss_url)
    return [gr.HTML.update(f"<div class=\"uploaded-image-box\"><img src={file}></img><div>", visible=True), file]
    
