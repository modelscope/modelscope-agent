import os
import random
import shutil
import traceback

import gradio as gr
import modelscope_gradio_components as mgr
from config_utils import get_avatar_image, get_ci_dir, parse_configuration
from gradio_utils import format_cover_html
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_gradio_components.components.Chatbot.llm_thinking_presets import \
    qwen
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
        logger.error(
            uuid=uuid_str,
            error=str(e),
            content={'error_traceback': traceback.format_exc()})
        raise Exception(e)
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
                user_chatbot = mgr.Chatbot(
                    value=[[None, '尝试问我一点什么吧～']],
                    elem_id='user_chatbot',
                    elem_classes=['markdown-body'],
                    avatar_images=avatar_pairs,
                    height=600,
                    latex_delimiters=[],
                    show_label=False,
                    show_copy_button=True,
                    llm_thinking_presets=[
                        qwen(
                            action_input_title='调用 <Action>',
                            action_output_title='完成调用')
                    ])
            with gr.Row():
                user_chatbot_input = mgr.MultimodalInput(
                    interactive=True,
                    placeholder='跟我聊聊吧～',
                    upload_button_props=dict(
                        file_count='multiple',
                        file_types=['file', 'image', 'audio', 'video',
                                    'text']))

        with gr.Column(scale=1):
            user_chat_bot_cover = gr.HTML(
                format_cover_html(builder_cfg, avatar_pairs[1]))
            user_chat_bot_suggest = gr.Examples(
                label='Prompt Suggestions',
                examples=suggests,
                inputs=[user_chatbot_input])

    def send_message(chatbot, input, _state):
        # 将发送的消息添加到聊天历史
        if 'user_agent' not in _state:
            init_user(_state)
        # 将发送的消息添加到聊天历史
        _uuid_str = check_uuid(uuid_str)
        user_agent = _state['user_agent']
        append_files = []
        for file in input.files:
            file_name = os.path.basename(file.path)
            # covert xxx.json to xxx_uuid_str.json
            file_name = file_name.replace('.', f'_{_uuid_str}.')
            file_path = os.path.join(get_ci_dir(), file_name)
            if not os.path.exists(file_path):
                # make sure file path's directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy(file.path, file_path)
            append_files.append(file_path)
        chatbot.append([{'text': input.text, 'files': input.files}, None])
        yield {
            user_chatbot: chatbot,
            user_chatbot_input: None,
        }

        response = ''
        try:
            for frame in user_agent.stream_run(
                    input.text,
                    print_info=True,
                    remote=False,
                    append_files=append_files):
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
                chatbot[-1][1] = response
                yield {
                    user_chatbot: chatbot,
                }
        except Exception as e:
            if 'dashscope.common.error.AuthenticationError' in str(e):
                msg = 'DASHSCOPE_API_KEY should be set via environment variable. You can acquire this in ' \
                    'https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key'
            elif 'rate limit' in str(e):
                msg = 'Too many people are calling, please try again later.'
            else:
                msg = str(e)
            chatbot[-1][1] = msg
            yield {user_chatbot: chatbot}

    gr.on([user_chatbot_input.submit],
          fn=send_message,
          inputs=[user_chatbot, user_chatbot_input, state],
          outputs=[user_chatbot, user_chatbot_input])

    demo.load(init_user, inputs=[state], outputs=[state])

demo.queue()
demo.launch(show_error=True, max_threads=10)
