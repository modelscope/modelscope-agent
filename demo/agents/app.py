import gradio as gr

# available models 
models = ["qwen-max", "qwen-plus"]

def update_preview(messages, preview_chat_input, name, description, instructions, conversation_starters, knowledge_files, capabilities_checkboxes):
    # 如果有文件被上传，获取所有文件名
    filenames = [f.name for f in knowledge_files] if knowledge_files else ["No files uploaded"]
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

def preview_send_message(message, chat_history):
    # 将发送的消息添加到聊天历史
    if message:
        chat_history.append(("You", message))
    return chat_history, ""  # 清空输入框的文本


def process_configuration(name, description, instructions, model, starters, files, capabilities_checkboxes):
    # 在这里处理配置逻辑，例如保存信息或更新系统配置
    # 当前只打印信息
    print(f"Name: {name}")
    print(f"Description: {description}")
    print(f"Instructions: {instructions}")
    print(f"Model: {model}")
    print(f"Conversation Starters: {starters}")
    print(f"Uploaded Files: {[f.name for f in files] if files else ['No files uploaded']}")
    print(f"capabilities_checkboxes: {capabilities_checkboxes}")
    return "配置已更新"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Create"):
                    with gr.Column():
                        # "Create" 标签页的 Chatbot 组件
                        create_chatbot = gr.Chatbot(label="Create Chatbot")
                        create_chat_input = gr.Textbox(label="Message", placeholder="Type a message here...")
                        create_send_button = gr.Button("Send")

                with gr.Tab("Configure"):
                    with gr.Column():
                        # "Configure" 标签页的配置输入字段
                        name_input = gr.Textbox(label="Name", placeholder="Name your GPT")
                        description_input = gr.Textbox(label="Description", placeholder="Add a short description about what this GPT does")
                        instructions_input = gr.Textbox(label="Instructions", placeholder="What does this GPT do? How does it behave? What should it avoid doing?")
                        model_selector = model_selector = gr.Dropdown(label='model', choices=models, value=models[0])
                        conversation_starters_input = gr.Textbox(label="Conversation starters", placeholder="Add conversation starters", lines=3)
                        #knowledge_input = gr.File(label="Knowledge", type="file", multiple=True)
                        knowledge_input = gr.File(label="Knowledge", file_count="multiple", file_types=["text", ".json", ".csv"])
                        capabilities_checkboxes = gr.CheckboxGroup(
                            label="Capabilities", 
                            choices=["Web Browsing", "Image Generation", "Code Interpreter"],
                            value=["Image Generation"]
                        )      
                        
                        with gr.Accordion("配置选项", open=False):
                            schema1 = gr.Textbox(label="Schema", placeholder="Enter your OpenAPI schema here")
                            auth1 = gr.Radio(label="Authentication", choices=["None", "API Key", "OAuth"])
                            privacy_policy1 = gr.Textbox(label="Privacy Policy", placeholder="Enter privacy policy URL")


                        configure_button = gr.Button("Update Configuration")

        with gr.Column():
            gr.Markdown(f"### Preview")

            # Preview
            preview_chatbot = gr.Chatbot(label="Chat History")
            preview_chat_input = gr.Textbox(label="Send a message", placeholder="Type a message...")
            preview_send_button = gr.Button("Send")
 
    # 配置 "Create" 标签页的消息发送功能
    create_send_button.click(
        create_send_message,
        inputs=[create_chatbot, create_chat_input],
        outputs=[create_chatbot, create_chat_input]
    )

    # 配置 "Configure" 标签页的提交按钮功能
    configure_button.click(
        process_configuration,
        inputs=[name_input, description_input, instructions_input, model_selector, conversation_starters_input, knowledge_input, capabilities_checkboxes],
        outputs=[]
    )

    # Preview 列消息发送
    preview_send_button.click(
        update_preview, 
        inputs=[preview_chatbot, preview_chat_input, name_input, description_input, instructions_input, conversation_starters_input, knowledge_input, capabilities_checkboxes], 
        outputs=[preview_chatbot, preview_chat_input]
    )

demo.launch()

