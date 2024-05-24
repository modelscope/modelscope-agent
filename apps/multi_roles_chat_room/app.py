import os
import re
from typing import Optional

import gradio as gr
import json
import modelscope_studio as mgr
from modelscope_agent.multi_agents_utils.executors.ray import RayTaskExecutor
from role_core import (chat_progress, init_all_remote_actors,
                       start_chat_with_topic)
from story_holder import get_avatar_by_name, get_story_by_id, stories

chat_history = []
RayTaskExecutor.init_ray()

# 发送消息的函数


def render_json_as_markdown(json_data):
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    markdown_str = '角色信息\n```json\n' + json_str + '\n```'
    print(markdown_str)
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
def upsert_user(new_user, user_char, _state):
    roles = _state['roles']
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


# end topic
def end_topic():
    chat_history.clear()
    return '', 'topic ended。'


def check_uuid(uid: Optional[str]) -> str:
    """Checks whether a UUID is provided or generates a default one."""
    if not uid or uid == '':
        if os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio':
            import gradio as gr

            raise gr.Error('Please login first')
        uid = 'local_user'
    return uid


def format_entry_html():
    base_html = '''
    <div class="container story-cardlist">
      <div class="row row-cols-1 row-cols-sm-2 row-cols-md-2 row-cols-lg-2 g-4">
    '''
    for card in stories:
        card_html = f'''
        <div class="col-md-4 gy-4" onclick="window.js_choose_story({card['id']})">
          <div class="card h-100">
            <img class="card-img-top" src={card['cover']}>
            <div class="card-body">
              <h5 class="card-title">{card['title']}</h5>
              <p class="card-text">{card['description']}</p>
            </div>
          </div>
        </div>
        '''
        base_html += card_html
    base_html += '''
      </div>
    </div>
    '''
    return base_html


# 创建Gradio界面
demo = gr.Blocks(
    css='assets/app.css',
    js='assets/app.js',
    head='''
    <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" crossorigin="anonymous"/>  # noqa E501
    ''')
with demo:
    state = gr.State({})
    story_state = gr.State('1')  # default value
    uuid = gr.Textbox(label='modelscope_uuid', visible=False)

    with gr.Column(visible=True) as entry:
        gr.Markdown('##  选择一个场景进入聊天吧～')
        entry_btn = gr.Button(
            elem_id='entry_fake_btn', visible=False, value='empty')
        gr.HTML(format_entry_html())

    with gr.Row(visible=False) as content:
        with gr.Column(scale=2):

            user_chatbot = mgr.Chatbot(
                value=[[None, None]],
                elem_id='user_chatbot',
                elem_classes=['markdown-body'],
                avatar_images=[None, None],
                avatar_image_width=60,
                height=650,
                show_label=True,
                visible=True,
                show_copy_button=True)
            preview_chat_input = mgr.MultimodalInput(
                interactive=False,
                label='输入',
                placeholder='输入你的消息',
                submit_button_props=dict(label='发送'))
        with gr.Column(scale=1):
            back_btn = gr.Button('返回重新选择场景')
            with gr.Group('Models'):
                select_model = gr.Dropdown(
                    label='The model that use as LLM core',
                    choices=list(['qwen-max', 'qwen-spark-plus-0403']),
                    value='qwen-spark-plus-0403')
            with gr.Group('Roles'):
                # new_user_name = gr.Textbox(
                #     label='Role name', placeholder='input role name ...')
                # new_user_char = gr.Textbox(
                #     label='Role characters',
                #     placeholder='input role characters ...')
                # new_user_btn = gr.Button('Add/Update role infos')
                # role_info = gr.Textbox(label='Result', interactive=False)
                all_roles = mgr.Markdown(
                    render_json_as_markdown(
                        get_story_by_id(story_state.value)['roles']))

            with gr.Group('Chat Room'):
                start_topic_from = gr.Dropdown(
                    label='The role who start the topic',
                    choices=list(
                        get_story_by_id(story_state.value)['roles'].keys()),
                    value=list(
                        get_story_by_id(story_state.value)['roles'].keys())[1])
                start_topic_input = gr.Textbox(
                    label='Topic to be discussed',
                    placeholder=get_story_by_id(
                        story_state.value)['default_topic'],
                    value=get_story_by_id(story_state.value)['default_topic'])
                user_select = gr.Dropdown(
                    label='Role playing',
                    choices=list(
                        get_story_by_id(story_state.value)['roles'].keys()),
                    value=list(
                        get_story_by_id(story_state.value)['roles'].keys())[0])
                start_chat_btn = gr.Button('Start new chat')
                # end_chat_btn = gr.Button("End chat")

    def choose_story(chosen_id):
        return {
            entry:
            gr.update(visible=False),
            content:
            gr.update(visible=True),
            story_state:
            chosen_id,
            all_roles:
            gr.update(
                value=render_json_as_markdown(
                    get_story_by_id(chosen_id)['roles'])),
            start_topic_from:
            gr.update(
                choices=list(get_story_by_id(chosen_id)['roles'].keys()),
                value=list(get_story_by_id(chosen_id)['roles'].keys())[1]),
            start_topic_input:
            gr.update(
                placeholder=get_story_by_id(chosen_id)['default_topic'],
                value=get_story_by_id(chosen_id)['default_topic']),
            user_select:
            gr.update(
                choices=list(get_story_by_id(chosen_id)['roles'].keys()),
                value=list(get_story_by_id(chosen_id)['roles'].keys())[0]),
        }

    entry_btn.click(
        fn=choose_story,
        inputs=[entry_btn],
        outputs=[
            entry, content, story_state, all_roles, start_topic_from,
            start_topic_input, user_select
        ],
        js='get_story_id')

    def back():
        return {
            entry: gr.update(visible=True),
            content: gr.update(visible=False),
        }

    back_btn.click(fn=back, inputs=[], outputs=[entry, content])

    def start_chat(username, from_user, topic, _state, _chatbot, _input,
                   _story_state, select_model, _uuid):
        # get uuid
        _uuid = check_uuid(_uuid)
        user_task_id = f'{_uuid[:6]}_{_story_state}'
        roles = get_story_by_id(_story_state)['roles']
        _state = init_all_remote_actors(roles, username, _state, _story_state,
                                        select_model, user_task_id)
        _state = start_chat_with_topic(from_user, topic, _state)

        init_chat = [{
            'avatar': get_avatar_by_name(from_user),
            'name': from_user,
            'text': topic
        }]
        _chatbot.append([None, init_chat])

        yield {state: _state, user_chatbot: _chatbot}

        bot_messages = {key: '' for key in _state['role_names']}

        new_round = False
        use_init = True

        for frame_text in chat_progress(None, _state):
            print(frame_text)
            role, content = get_frame_data(frame_text)
            if role in bot_messages:
                bot_messages[role] += content
                output = []
                for item in bot_messages:
                    if bot_messages[item] != '':
                        output.append({
                            'avatar': get_avatar_by_name(item),
                            'name': item,
                            'text': bot_messages[item]
                        })

                if not new_round:
                    if use_init:
                        _chatbot[-1][1] = init_chat + output
                    else:
                        _chatbot[-1][1] = output
                else:
                    _chatbot.append([None, output])
                    new_round = False
                yield {
                    user_chatbot: _chatbot,
                }

            if frame_text == 'new_round':
                new_round = True
                use_init = False
                bot_messages = {key: '' for key in _state['role_names']}
                continue

            # try to parse the next_speakers from yield result
            try:
                next_speakers = json.loads(frame_text)['next_agent_names']
                _state['next_agent_names'] = next_speakers
                yield {
                    state: _state,
                    preview_chat_input:
                    gr.update(interactive=True, value=None),
                }
            except Exception:
                pass

    def send_message(_chatbot, _input, _state):
        _chatbot.append([_input.text, None])
        yield {
            state: _state,
            preview_chat_input: gr.update(interactive=False, value=None),
            user_chatbot: mgr.Chatbot(visible=True, value=_chatbot)
        }

        bot_messages = {key: '' for key in _state['role_names']}
        new_round = False

        for frame_text in chat_progress(_input.text, _state):
            role, content = get_frame_data(frame_text)
            if role in bot_messages:
                bot_messages[role] += content
                output = []
                for item in bot_messages:
                    if bot_messages[item] != '':
                        output.append({
                            'avatar': get_avatar_by_name(item),
                            'name': item,
                            'text': bot_messages[item]
                        })

                if not new_round:
                    _chatbot[-1][1] = output
                else:
                    _chatbot.append([None, output])
                    new_round = False
                yield {
                    user_chatbot: _chatbot,
                }

            if frame_text == 'new_round':
                new_round = True
                bot_messages = {key: '' for key in _state['role_names']}
                continue

            # try to parse the next_speakers from yield result
            try:
                next_agent_names = json.loads(frame_text)['next_agent_names']
                _state['next_agent_names'] = next_agent_names
                yield {
                    state: _state,
                    preview_chat_input:
                    gr.update(interactive=True, value=None),
                }
            except Exception:
                pass

    # send message btn
    preview_chat_input.submit(
        fn=send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, preview_chat_input, state])

    # add or update role btn
    # new_user_btn.click(
    #     fn=upsert_user,
    #     inputs=[new_user_name, new_user_char, state],
    #     outputs=[user_select, role_info, all_roles])

    # start new chat btn
    start_chat_btn.click(
        fn=start_chat,
        inputs=[
            user_select, start_topic_from, start_topic_input, state,
            user_chatbot, preview_chat_input, story_state, select_model, uuid
        ],
        outputs=[user_chatbot, preview_chat_input, state])

demo.launch()
