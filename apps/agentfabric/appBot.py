import random
import sys
import traceback

import gradio as gr
from config_utils import get_avatar_image, parse_configuration
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
                    preview_send_button = gr.Button('发送', variant='primary')

        with gr.Column(scale=1):
            user_chat_bot_cover = gr.HTML(
                format_cover_html(builder_cfg, avatar_pairs[1]))
            user_chat_bot_suggest = gr.Examples(
                label='Prompt Suggestions',
                examples=suggests,
                inputs=[preview_chat_input])

    def send_message(chatbot, input, _state):
        # 将发送的消息添加到聊天历史
        user_agent = _state['user_agent']
        chatbot.append((input, ''))
        yield {
            user_chatbot: chatbot,
            preview_chat_input: gr.Textbox.update(value=''),
        }

        response = ''

        for frame in user_agent.stream_run(
                input, print_info=True, remote=False):
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

    preview_send_button.click(
        send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, preview_chat_input])

    demo.load(init_user, inputs=[state], outputs=[state])

demo.queue()
demo.launch()
