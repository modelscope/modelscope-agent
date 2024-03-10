import re

import gradio as gr
import json
import modelscope_studio as mgr
from role_core import (chat_progress, init_all_remote_actors, roles,
                       start_chat_with_topic)

chat_history = []


# 发送消息的函数
def send_message(_chatbot, _input, _state):
    _chatbot.append([_input.text, None])
    yield {
        state: _state,
        preview_chat_input: gr.update(interactive=False, value=None),
        user_chatbot: mgr.Chatbot(visible=True, value=_chatbot)
    }

    bot_messages = {key: '' for key in _state['role_names']}

    for frame_text in chat_progress(_chatbot.text, _state):
        role, content = get_frame_data(frame_text)
        if role in bot_messages:
            bot_messages[role] += content
        _chatbot[-1][1] = [bot_messages[key] for key in _state['role_names']]

        # try to parse the next_speakers from yield result
        try:
            next_speakers = json.loads(frame_text)['next_speakers']
            _state['next_agent_names'] = next_speakers
            yield {state: _state}
        except Exception:
            pass

    yield {
        user_chatbot: _chatbot,
    }


def render_json_as_markdown(json_data):
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    markdown_str = '角色信息\n```json\n' + json_str + '\n```'
    return markdown_str


def get_frame_data(text):
    pattern = r'<([^>]+)>: (.*)'
    # 使用正则表达式搜索文本
    match = re.search(pattern, text)

    # 如果找到匹配项，则提取所需的部分
    if match:
        role = match.group(1)  # 尖括号内的字符
        content = match.group(2)  # 尖括号之后的字符
        return role, content
    else:
        return None, None


# update or add user
def upsert_user(new_user, user_char):

    if new_user and new_user not in roles:
        roles[new_user] = user_char
        return gr.update(
            choices=roles), f'User {new_user} added', render_json_as_markdown(
                roles)
    else:
        roles[new_user] = user_char
        return gr.update(
            choices=roles
        ), f'User {new_user} updated', render_json_as_markdown(roles)


# start topic
def start_chat(username, from_user, topic, _state, _chatbot, _input):
    _state = init_all_remote_actors(roles, username, _state)
    _state = start_chat_with_topic(from_user, topic, _state)

    _chatbot.append([topic, None])
    yield {
        state: _state,
        preview_chat_input: gr.update(interactive=False, value=None),
        user_chatbot: _chatbot
    }

    bot_messages = {key: '' for key in _state['role_names']}

    for frame_text in chat_progress(None, _state):
        role, content = get_frame_data(frame_text)
        if role in bot_messages:
            bot_messages[role] += content
        _chatbot[-1][1] = [bot_messages[key] for key in _state['role_names']]

        # try to parse the next_speakers from yield result
        try:
            next_speakers = json.loads(frame_text)['next_speakers']
            _state['next_agent_names'] = next_speakers
            yield {state: _state}
        except Exception:
            pass

    yield {
        user_chatbot: _chatbot,
    }


# end topic
def end_topic():
    chat_history.clear()
    return '', 'topic ended。'


# 创建Gradio界面
with gr.Blocks() as demo:
    state = gr.State({})
    with gr.Row():
        with gr.Column(scale=2):
            user_chatbot = mgr.Chatbot(
                value=[[None, None]],
                elem_id='user_chatbot',
                elem_classes=['markdown-body'],
                avatar_images=[None],
                height=650,
                show_label=False,
                visible=False,
                show_copy_button=True)
            preview_chat_input = mgr.MultimodalInput(
                interactive=False,
                label='输入',
                placeholder='输入你的消息',
                submit_button_props=dict(label='发送（role 加载中...）'))
        with gr.Column(scale=1):
            with gr.Group('Roles'):
                new_user_name = gr.Textbox(
                    label='Role name', placeholder='input role name ...')
                new_user_char = gr.Textbox(
                    label='Role characters',
                    placeholder='input role characters ...')
                new_user_btn = gr.Button('Add/Update role infos')
                role_info = gr.Textbox(label='Result', interactive=False)
                all_roles = gr.Markdown(render_json_as_markdown(roles))

            with gr.Group('Chat Room'):
                start_topic_from = gr.Dropdown(
                    label='The role who start the topic',
                    choices=list(roles.keys()),
                    value=list(roles.keys())[1])
                start_topic_input = gr.Textbox(
                    label='Topic to be discussed',
                    placeholder='@顾易 要不要来我家吃饭？',
                    value='@顾易 要不要来我家吃饭？')
                user_select = gr.Dropdown(
                    label='Role playing',
                    choices=list(roles.keys()),
                    value=list(roles.keys())[0])
                start_chat_btn = gr.Button('Start new chat')
                # end_chat_btn = gr.Button("End chat")

    input = mgr.MultimodalInput()

    # send message btn
    preview_chat_input.submit(
        fn=send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, preview_chat_input, state])

    # add or update role btn
    new_user_btn.click(
        fn=upsert_user,
        inputs=[new_user_name, new_user_char],
        outputs=[user_select, role_info, all_roles])

    # start new chat btn
    start_chat_btn.click(
        fn=start_chat,
        inputs=[
            user_select, start_topic_from, start_topic_input, state,
            user_chatbot, preview_chat_input
        ],
        outputs=[user_chatbot, preview_chat_input, role_info, state])

    # # end chat btn
    # end_chat_btn.click(fn=end_topic, inputs=[], outputs=[chat_box, role_info])

demo.launch()
