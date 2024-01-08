from __future__ import annotations
import os
import re
import traceback
import uuid
from copy import deepcopy

import gradio as gr
import json


def stream_predict(
        chatbot,  # ChatBot
        user_input,  # Textbox
        upload_image_url,  # imageUrl
        agent  # agent
):
    print(f'upload_image_url: {upload_image_url}')
    if not user_input:
        if len(history) == 0:
            yield chatbot, '请输入问题……'
            return
        else:
            user_input = chatbot[-1][0]
            chatbot = chatbot[:-1]
            yield chatbot, '重新生成回答……'
    else:
        print(upload_image_url)
        if upload_image_url:
            user_input = f'![upload]({upload_image_url})\n' + user_input
        print('user_input:', user_input)
        yield chatbot, '开始生成回答……'

    chatbot.append((user_input, '处理中...'))

    yield chatbot, '开始实时传输回答……'

    response = ''
    try:
        for frame in agent.stream_run(user_input, remote=False):
            is_final = frame.get('is_final')
            llm_result = frame.get('llm_text', '')
            exec_result = frame.get('exec_result', '')
            error_message = frame.get('error')
            if is_final:
                chatbot[-1] = (chatbot[-1][0], response)
                yield chatbot, '完成回答'
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
    except Exception:
        traceback.print_exc()
        chatbot[-1] = (chatbot[-1][0], 'chat_async error.')
        yield chatbot, ''


def upload_image(file):

    gr_file_path = f'./file={file.name}'

    return [
        gr.HTML.update(
            f"<div class=\"uploaded-image-box\"><img src=\"{gr_file_path}\"></img><div>",
            visible=True), gr_file_path
    ]
