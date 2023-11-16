import sys
import traceback

import gradio as gr
from config_utils import parse_configuration, save_builder_configuration
from gradio_utils import ChatBot
from user_core import init_user_chatbot_agent

sys.path.append('../')


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


def init_ui_config(state, builder_cfg, model_cfg, tool_cfg):
    print("builder_cfg:", builder_cfg)
    # available models
    models = list(model_cfg.keys())
    capabilities = [(tool_cfg[tool_key]["name"], tool_key)
                    for tool_key in tool_cfg.keys()
                    if tool_cfg[tool_key].get("is_active", False)]
    state["tool_cfg"] = tool_cfg
    state["capabilities"] = capabilities
    suggests = builder_cfg.get("suggests", [])
    return [
        state,
        # config form
        builder_cfg.get('name', ''),
        builder_cfg.get('description'),
        builder_cfg.get('instruction'),
        gr.Dropdown.update(
            value=builder_cfg.get("model", models[0]), choices=models),
        [[str] for str in suggests],
        builder_cfg.get("knowledge", [])
        if len(builder_cfg["knowledge"]) > 0 else None,
        gr.CheckboxGroup.update(
            value=[
                tool for tool in builder_cfg.get("tools", {}).keys()
                if builder_cfg.get("tools").get(tool).get("use", False)
            ],
            choices=capabilities),
        # bot
        format_cover_html(builder_cfg),
        gr.Dataset.update(samples=[[item] for item in suggests]),
    ]


def init_all(state):
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
    )
    ret = init_ui_config(state, builder_cfg, model_cfg, tool_cfg)
    yield ret
    init_user(state)
    yield ret


def reset_agent(state):
    user_agent = state['user_agent']
    user_agent.reset()
    state['user_agent'] = user_agent


def format_cover_html(configuration):
    return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src="//img.alicdn.com/imgextra/i3/O1CN01YPqZFO1YNZerQfSBk_!!6000000003047-0-tps-225-225.jpg" />
    </div>
    <div class="bot_name">{configuration.get("name", "")}</div>
    <div class="bot_desp">{configuration.get("description", "")}</div>
</div>
"""


def format_preview_send_message_ret(preview_chatbot):
    return [
        gr.Chatbot.update(visible=True, value=preview_chatbot),
        gr.HTML.update(visible=False)
    ]


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
        # llm_result = llm_result.split("<|user|>")[0].strip()
        if len(exec_result) != 0:
            # action_exec_result
            if isinstance(exec_result, dict):
                exec_result = str(exec_result['result'])
            frame_text = f'Result: {exec_result}'
        else:
            # llm result
            frame_text = llm_result
        response = f'{response}\n{frame_text}'
        preview_chatbot[-1] = (preview_chat_input, response)
        yield format_preview_send_message_ret(preview_chatbot)


def process_configuration(name, description, instructions, model, suggestions,
                          files, capabilities_checkboxes, state):
    tool_cfg = state["tool_cfg"]
    capabilities = state["capabilities"]

    builder_cfg = {
        "name": name,
        "avatar": "",
        "description": description,
        "instruction": instructions,
        "suggests": [row[0] for row in suggestions],
        "knowledge": list(map(lambda file: file.name, files or [])),
        "tools": {
            capability: dict(
                name=tool_cfg[capability]["name"],
                is_active=tool_cfg[capability]["is_active"],
                use=True if capability in capabilities_checkboxes else False)
            for capability in map(lambda item: item[1], capabilities)
        },
        "model": model,
    }
    save_builder_configuration(builder_cfg)
    init_user(state)
    return [
        gr.HTML.update(visible=True, value=format_cover_html(builder_cfg)),
        gr.Chatbot.update(visible=False),
        gr.Dataset.update(samples=suggestions)
    ]


# 创建 Gradio 界面
demo = gr.Blocks(css="assets/app.css")
with demo:
    state = gr.State({})

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
                            label="Name", placeholder="Name your GPT")
                        description_input = gr.Textbox(
                            label="Description",
                            placeholder=
                            "Add a short description about what this GPT does")
                        instructions_input = gr.Textbox(
                            label="Instructions",
                            placeholder=
                            "What does this GPT do? How does it behave? What should it avoid doing?",
                            lines=3)
                        model_selector = model_selector = gr.Dropdown(
                            label='model')
                        suggestion_input = gr.Dataframe(
                            show_label=False,
                            value=[['']],
                            datatype=["str"],
                            headers=['prompt suggestion'],
                            type="array",
                            col_count=(1, "fixed"),
                            interactive=True)
                        knowledge_input = gr.File(
                            label="Knowledge",
                            file_count="multiple",
                            file_types=["text", ".json", ".csv"])
                        capabilities_checkboxes = gr.CheckboxGroup(
                            label="Capabilities")

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
            # Preview
            gr.HTML("""<div class="preview_header">Preview<div>""")

            user_chat_bot_cover = gr.HTML(format_cover_html({}))
            user_chatbot = ChatBot(
                value=[[None, None]],
                elem_id="user_chatbot",
                elem_classes=["markdown-body"],
                latex_delimiters=[],
                show_label=False,
                visible=False)
            preview_chat_input = gr.Textbox(
                label="Send a message", placeholder="Type a message...")
            user_chat_bot_suggest = gr.Dataset(
                label="Prompt Suggestions",
                components=[preview_chat_input],
                samples=[])
            preview_send_button = gr.Button("Send")
            user_chat_bot_suggest.select(
                lambda evt: evt[0],
                inputs=[user_chat_bot_suggest],
                outputs=[preview_chat_input])

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
            suggestion_input, knowledge_input, capabilities_checkboxes, state
        ],
        outputs=[user_chat_bot_cover, user_chatbot, user_chat_bot_suggest])

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

    demo.load(
        init_all,
        inputs=[state],
        outputs=[
            state,
            # config form
            name_input,
            description_input,
            instructions_input,
            model_selector,
            suggestion_input,
            knowledge_input,
            capabilities_checkboxes,
            # bot
            user_chat_bot_cover,
            user_chat_bot_suggest,
        ])

demo.queue()
demo.launch()
