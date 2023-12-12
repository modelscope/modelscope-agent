import os
import random
import shutil
import sys
import traceback

import gradio as gr
from config_utils import get_avatar_image, get_ci_dir, parse_configuration
from gradio_utils import ChatBot, format_cover_html
from user_core import init_user_chatbot_agent

uuid_str = 'local_user'
builder_cfg, model_cfg, tool_cfg, available_tool_list, _, _ = parse_configuration(
    uuid_str)
suggests = builder_cfg.get('prompt_recommend', [])
avatar_pairs = get_avatar_image(builder_cfg.get('avatar', ''), uuid_str)

customTheme = gr.themes.Default(
    primary_hue=gr.themes.utils.colors.blue,
    radius_size=gr.themes.utils.sizes.radius_none,
)


def check_uuid(uuid_str):
    if not uuid_str or uuid_str == '':
        if os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio':
            raise gr.Error('请登陆后使用! (Please login first)')
        else:
            uuid_str = 'local_user'
    return uuid_str


def init_user(state):
    try:
        seed = state.get('session_seed', random.randint(0, 1000000000))
        user_agent = init_user_chatbot_agent(uuid_str)
        user_agent.seed = seed
        state['user_agent'] = user_agent
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    return state


# 创建 Gradio 界面
demo = gr.Blocks(css='assets/appBot.css', theme=customTheme)
with demo:
    gr.Markdown(
        '# <center> \N{fire} AgentFabric powered by Modelscope-agent ([github star](https://github.com/modelscope/modelscope-agent/tree/main))</center>'  # noqa E501
    )
    draw_seed = random.randint(0, 1000000000)
    state = gr.State({'session_seed': draw_seed})
    with gr.Row(elem_classes='container'):
        with gr.Column(scale=4):
            with gr.Column():
                # Preview
                user_chatbot = ChatBot(
                    value=[[None, '尝试问我一点什么吧～']],
                    elem_id='user_chatbot',
                    elem_classes=['markdown-body'],
                    avatar_images=avatar_pairs,
                    height=600,
                    latex_delimiters=[],
                    show_label=False)
            with gr.Row():
                with gr.Column(scale=12):
                    preview_chat_input = gr.Textbox(
                        show_label=False,
                        container=False,
                        placeholder='跟我聊聊吧～')
                with gr.Column(min_width=70, scale=1):
                    upload_button = gr.UploadButton(
                        '上传',
                        file_types=['file', 'image', 'audio', 'video', 'text'],
                        file_count='multiple')
                with gr.Column(min_width=70, scale=1):
                    preview_send_button = gr.Button('发送', variant='primary')

        with gr.Column(scale=1):
            user_chat_bot_cover = gr.HTML(
                format_cover_html(builder_cfg, avatar_pairs[1]))
            user_chat_bot_suggest = gr.Examples(
                label='Prompt Suggestions',
                examples=suggests,
                inputs=[preview_chat_input])

    def upload_file(chatbot, upload_button, _state):
        _uuid_str = check_uuid(uuid_str)
        new_file_paths = []
        if 'file_paths' in _state:
            file_paths = _state['file_paths']
        else:
            file_paths = []
        for file in upload_button:
            file_name = os.path.basename(file.name)
            # covert xxx.json to xxx_uuid_str.json
            file_name = file_name.replace('.', f'_{_uuid_str}.')
            file_path = os.path.join(get_ci_dir(), file_name)
            if not os.path.exists(file_path):
                # make sure file path's directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy(file.name, file_path)
                file_paths.append(file_path)
            new_file_paths.append(file_path)
            if file_name.endswith(('.jpeg', '.png', '.jpg')):
                chatbot += [((file_path, ), None)]

            else:
                chatbot.append((None, f'上传文件{file_name}，成功'))
        yield {
            user_chatbot: gr.Chatbot.update(visible=True, value=chatbot),
            preview_chat_input: gr.Textbox.update(value='')
        }

        _state['file_paths'] = file_paths
        _state['new_file_paths'] = new_file_paths

    upload_button.upload(
        upload_file,
        inputs=[user_chatbot, upload_button, state],
        outputs=[user_chatbot, preview_chat_input])

    def send_message(chatbot, input, _state):
        # 将发送的消息添加到聊天历史
        user_agent = _state['user_agent']
        if 'new_file_paths' in _state:
            new_file_paths = _state['new_file_paths']
        else:
            new_file_paths = []
        _state['new_file_paths'] = []
        chatbot.append((input, ''))
        yield {
            user_chatbot: chatbot,
            preview_chat_input: gr.Textbox.update(value=''),
        }

        response = ''
        try:
            for frame in user_agent.stream_run(
                    input,
                    print_info=True,
                    remote=False,
                    append_files=new_file_paths):
                # is_final = frame.get("frame_is_final")
                llm_result = frame.get('llm_text', '')
                exec_result = frame.get('exec_result', '')
                # llm_result = llm_result.split("<|user|>")[0].strip()
                if len(exec_result) != 0:
                    # action_exec_result
                    if isinstance(exec_result, dict):
                        exec_result = str(exec_result['result'])
                    frame_text = f'<result>{exec_result}</result>'
                else:
                    # llm result
                    frame_text = llm_result

                # important! do not change this
                response += frame_text
                chatbot[-1] = (input, response)
                yield {
                    user_chatbot: chatbot,
                }
        except Exception as e:
            if 'dashscope.common.error.AuthenticationError' in str(e):
                msg = 'DASHSCOPE_API_KEY should be set via environment variable. You can acquire this in ' \
                    'https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key'
            else:
                msg = str(e)
            chatbot[-1] = (input, msg)
            yield {user_chatbot: chatbot}

    preview_send_button.click(
        send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, preview_chat_input])

    demo.load(init_user, inputs=[state], outputs=[state])

demo.queue()
demo.launch()
