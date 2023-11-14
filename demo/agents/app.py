import traceback

import gradio as gr
from builder_core import execute_user_chatbot, parse_configuration, init_user_chatbot_agent

from gradio_utils import ChatBot

# available models
models = ["qwen-max", "qwen-plus"]

builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration()

def format_cover_html(configuration):
    print('configuration:', configuration)
    return f"""
<div class="bot_cover">
    <div class="bot_avatar"><img src="//img.alicdn.com/imgextra/i3/O1CN01YPqZFO1YNZerQfSBk_!!6000000003047-0-tps-225-225.jpg" /></div>
    <div class="bot_name">{configuration["name"]}</div>
    <div class="bot_desp">{configuration["description"]}</div>
</div>
"""

def update_preview(messages, preview_chat_input, name, description,
                   instructions, conversation_starters, knowledge_files,
                   capabilities_checkboxes):
    # 如果有文件被上传，获取所有文件名
    filenames = [f.name for f in knowledge_files
                 ] if knowledge_files else ["No files uploaded"]
    # 模拟生成预览，这里只是简单地返回输入的文本和选择的功能
    preview_text = f"Name: {name}\n"
    preview_text += f"Description: {description}\n"
    preview_text += f"Conversation starters: {', '.join(conversation_starters)}\n"
    preview_text += f"knowledge files: {', '.join(filenames)}\n"
    preview_text += f"Capabilities: {', '.join(capabilities_checkboxes)}\n"
    messages.append((preview_chat_input, preview_text))
    return messages, preview_chat_input


def create_send_message(messages, message):
    # 模拟发送消息
    messages.append(("You", message))
    # 假设这里有一个生成响应的逻辑
    bot_response = "这是模拟的回应。"
    messages.append(("Bot", bot_response))
    return messages, ""


def init_user(state):
    try:
        user_agent = init_user_chatbot_agent()
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    state['user_agent'] = user_agent


def init_builder(state):
    try:
        builder_agent = init_builder_chatbot_agent()
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    state['builder_agent'] = builder_agent


def reset_agent(state):
    user_agent = state['user_agent']
    user_agent.reset()
    state['user_agent'] = user_agent


def format_preview_send_message_ret(preview_chatbot):
    return [gr.Chatbot.update(visible=True, value=preview_chatbot), gr.HTML.update(visible=False)]
    
def preview_send_message(preview_chatbot, preview_chat_input, state):
    # 将发送的消息添加到聊天历史
    user_agent = state['user_agent']
    preview_chatbot.append((preview_chat_input, ""))
    yield format_preview_send_message_ret(preview_chatbot)

    response = ''

    for frame in user_agent.stream_run(
            preview_chat_input, print_info=True, remote=False):
        # is_final = frame.get("frame_is_final")
        llm_result = frame.get("llm_text", "")
        exec_result = frame.get('exec_result', '')
        print(frame)
        # llm_result = llm_result.split("<|user|>")[0].strip()
        if len(exec_result) != 0:
            # action_exec_result
            if isinstance(exec_result, dict):
                exec_result = str(exec_result['result'])
            frame_text = f'\n\n<|startofexec|>\n{exec_result}\n<|endofexec|>\n'
        else:
            # llm result
            frame_text = llm_result
        response = f'{response}\n{frame_text}'
        preview_chatbot[-1] = (preview_chat_input, response)
        yield format_preview_send_message_ret(preview_chatbot)


def process_configuration(name, description, instructions, model, starters,
                          files, capabilities_checkboxes):
    # 在这里处理配置逻辑，例如保存信息或更新系统配置
    # 当前只打印信息
    print(f"Name: {name}")
    print(f"Description: {description}")
    print(f"Instructions: {instructions}")
    print(f"Model: {model}")
    print(f"Conversation Starters: {starters}")
    print(
        f"Uploaded Files: {[f.name for f in files] if files else ['No files uploaded']}"
    )
    print(f"capabilities_checkboxes: {capabilities_checkboxes}")
    configuration = {
        "name": name,
        "description": description,
    }
    return [format_cover_html(configuration)]


# 创建 Gradio 界面
with gr.Blocks(css="assets/app.css") as demo:
    state = gr.State({})
    demo.load(init_user, inputs=[state], outputs=[])
    demo.load(init_builder, inputs=[state], outputs=[])

    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Create"):
                    with gr.Column():
                        # "Create" 标签页的 Chatbot 组件
                        create_chatbot = gr.Chatbot(label="Create Chatbot")
                        create_chat_input = gr.Textbox(
                            label="Message",
                            placeholder="Type a message here...")
                        create_send_button = gr.Button("Send")

                with gr.Tab("Configure"):
                    with gr.Column():
                        # "Configure" 标签页的配置输入字段
                        name_input = gr.Textbox(
                            label="Name", placeholder="Name your GPT", value=builder_cfg["name"])
                        description_input = gr.Textbox(
                            label="Description",
                            placeholder=
                            "Add a short description about what this GPT does", value=builder_cfg["description"])
                        instructions_input = gr.Textbox(
                            label="Instructions",
                            placeholder=
                            "What does this GPT do? How does it behave? What should it avoid doing?",
                            value=builder_cfg["instruction"]
                        )
                        model_selector = model_selector = gr.Dropdown(
                            label='model', choices=models, value=models[0])
                        conversation_starters_input = gr.Textbox(
                            label="Conversation starters",
                            placeholder="Add conversation starters",
                            lines=3)
                        knowledge_input = gr.File(
                            label="Knowledge",
                            file_count="multiple",
                            file_types=["text", ".json", ".csv"])
                        capabilities_checkboxes = gr.CheckboxGroup(
                            label="Capabilities",
                            choices=[
                                "Web Browsing", "Image Generation",
                                "Code Interpreter"
                            ],
                            value=["Image Generation"])

                        with gr.Accordion("配置选项", open=False):
                            schema1 = gr.Textbox(
                                label="Schema",
                                placeholder="Enter your OpenAPI schema here")
                            auth1 = gr.Radio(
                                label="Authentication",
                                choices=["None", "API Key", "OAuth"])
                            privacy_policy1 = gr.Textbox(
                                label="Privacy Policy",
                                placeholder="Enter privacy policy URL")

                        configure_button = gr.Button("Update Configuration")

        with gr.Column():
            gr.HTML("""<div class="preview_header">Preview<div>""")

            user_chat_bot_cover = gr.HTML(value=format_cover_html(builder_cfg))
            # Preview
            user_chatbot = ChatBot(
                value=[[None, None]],
                elem_id="user_chatbot",
                elem_classes=["markdown-body"],
                latex_delimiters=[],

                show_label=False,
                visible=False
            )
            preview_chat_input = gr.Textbox(
                label="Send a message", placeholder="Type a message...")
            preview_send_button = gr.Button("Send")

    # 配置 "Create" 标签页的消息发送功能
    create_send_button.click(
        create_send_message,
        inputs=[create_chatbot, create_chat_input],
        outputs=[create_chatbot, create_chat_input])

    # 配置 "Configure" 标签页的提交按钮功能
    configure_button.click(
        process_configuration,
        inputs=[
            name_input, description_input, instructions_input, model_selector,
            conversation_starters_input, knowledge_input,
            capabilities_checkboxes
        ],
        outputs=[user_chat_bot_cover])

    # Preview 列消息发送
    # preview_send_button.click(
    #     lambda _: [gr.Chatbot.update(visible=True), gr.HTML.update(visible=False)],
    #     inputs=[],
    #     outputs=[user_chatbot, user_chat_bot_cover]
    # )
    preview_send_button.click(
        preview_send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, user_chat_bot_cover])

demo.queue(max_size=1)
demo.launch()
