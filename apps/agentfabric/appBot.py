import os
import random
import shutil
import traceback

import gradio as gr
import modelscope_studio as mgr
from config_utils import get_avatar_image, get_ci_dir, parse_configuration
from gradio_utils import format_cover_html
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_studio.components.Chatbot.llm_thinking_presets import qwen
from user_core import init_user_chatbot_agent

uuid_str = 'local_user'
builder_cfg, model_cfg, tool_cfg, available_tool_list, _, _ = parse_configuration(
    uuid_str)
prologue = builder_cfg.get('prologue', '尝试问我一点什么吧～')
suggests = builder_cfg.get('prompt_recommend', [])
avatar_pairs = get_avatar_image(builder_cfg.get('avatar', ''), uuid_str)

customTheme = gr.themes.Default(
    primary_hue=gr.themes.utils.colors.blue,
    radius_size=gr.themes.utils.sizes.radius_none,
)

shutil.rmtree("/tmp/agentfabric/config/local_user/", ignore_errors=True)

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
        user_agent, user_memory = init_user_chatbot_agent(uuid_str)
        user_agent.seed = seed
        state['user_agent'] = user_agent
        state['user_memory'] = user_memory
    except Exception as e:
        logger.query_error(
            uuid=uuid_str,
            error=str(e),
            details={'error_traceback': traceback.format_exc()})
        raise Exception(e)
    return state


# 创建 Gradio 界面
demo = gr.Blocks(css='assets/appBot.css', theme=customTheme)
with demo:
    gr.Markdown(
        '# <center class="agent_title"> \N{fire} AgentFabric powered by Modelscope-agent [github star](https://github.com/modelscope/modelscope-agent/tree/main)</center>'  # noqa E501
    )
    draw_seed = random.randint(0, 1000000000)
    state = gr.State({'session_seed': draw_seed})
    with gr.Row(elem_classes='container'):
        with gr.Column(scale=4):
            with gr.Column():
                # Preview
                user_chatbot = mgr.Chatbot(
                    value=[[None, prologue]],
                    elem_id='user_chatbot',
                    elem_classes=['markdown-body'],
                    avatar_images=avatar_pairs,
                    height=600,
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
        user_memory = _state['user_memory']
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

        # get short term memory history
        history = user_memory.get_history()

        # get long term memory knowledge, currently get one file
        uploaded_file = None
        if len(append_files) > 0:
            uploaded_file = append_files[0]
        ref_doc = user_memory.run(
            query=input.text, url=uploaded_file, checked=True)

        response = ''
        try:
            for frame in user_agent.run(
                    input.text,
                    history=history,
                    ref_doc=ref_doc,
                    append_files=append_files):

                # important! do not change this
                response += frame
                chatbot[-1][1] = response
                yield {
                    user_chatbot: chatbot,
                }
            if len(history) == 0:
                user_memory.update_history(
                    Message(role='system', content=user_agent.system_prompt))

            user_memory.update_history([
                Message(role='user', content=input.text),
                Message(role='assistant', content=response),
            ])
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
