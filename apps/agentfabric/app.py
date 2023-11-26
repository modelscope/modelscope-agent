import os
import random
import traceback

import gradio as gr
import json
from builder_core import init_builder_chatbot_agent
from config_utils import (Config, get_avatar_image, get_user_cfg_file,
                          parse_configuration, save_avatar_image,
                          save_builder_configuration)
from gradio_utils import ChatBot, format_cover_html
from user_core import init_user_chatbot_agent


def init_user(uuid_str, state):
    try:
        seed = state.get('session_seed', random.randint(0, 1000000000))
        user_agent = init_user_chatbot_agent(uuid_str)
        user_agent.seed = seed
        state['user_agent'] = user_agent
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    return state


def init_builder(uuid_str, state):

    try:
        builder_agent = init_builder_chatbot_agent(uuid_str)
        state['builder_agent'] = builder_agent
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    return state


def update_builder(uuid_str, state):
    builder_agent = state['builder_agent']

    try:
        builder_cfg_file = get_user_cfg_file(uuid_str=uuid_str)
        with open(builder_cfg_file, 'r') as f:
            config = json.load(f)
        builder_agent.update_config_to_history(config)
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    return state


def init_ui_config(uuid_str, state, builder_cfg, model_cfg, tool_cfg):
    print('builder_cfg:', builder_cfg)
    # available models
    models = list(model_cfg.keys())
    capabilities = [(tool_cfg[tool_key]['name'], tool_key)
                    for tool_key in tool_cfg.keys()
                    if tool_cfg[tool_key].get('is_active', False)]
    state['model_cfg'] = model_cfg
    state['tool_cfg'] = tool_cfg
    state['capabilities'] = capabilities
    bot_avatar = get_avatar_image(builder_cfg.get('avatar', ''), uuid_str)[1]
    suggests = builder_cfg.get('conversation_starters', [])
    return [
        state,
        # config form
        gr.Image.update(value=bot_avatar),
        builder_cfg.get('name', ''),
        builder_cfg.get('description'),
        builder_cfg.get('instruction'),
        gr.Dropdown.update(
            value=builder_cfg.get('model', models[0]), choices=models),
        [[str] for str in suggests],
        builder_cfg.get('knowledge', [])
        if len(builder_cfg['knowledge']) > 0 else None,
        gr.CheckboxGroup.update(
            value=[
                tool for tool in builder_cfg.get('tools', {}).keys()
                if builder_cfg.get('tools').get(tool).get('use', False)
            ],
            choices=capabilities),
        # bot
        format_cover_html(builder_cfg, bot_avatar),
        gr.Dataset.update(samples=[[item] for item in suggests]),
    ]


def init_all(uuid_str, state):
    uuid_str = check_uuid(uuid_str)
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
        uuid_str)
    ret = init_ui_config(uuid_str, state, builder_cfg, model_cfg, tool_cfg)
    yield ret
    init_user(uuid_str, state)
    init_builder(uuid_str, state)
    yield ret


def check_uuid(uuid_str):
    if not uuid_str or uuid_str == '':
        if os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio':
            raise gr.Error('请登陆后使用! (Please login first)')
        else:
            uuid_str = 'local_user'
    return uuid_str


def process_configuration(uuid_str, bot_avatar, name, description,
                          instructions, model, suggestions, files,
                          capabilities_checkboxes, state):
    uuid_str = check_uuid(uuid_str)
    tool_cfg = state['tool_cfg']
    capabilities = state['capabilities']
    bot_avatar, bot_avatar_path = save_avatar_image(bot_avatar, uuid_str)
    suggestions_filtered = [row for row in suggestions if row[0]]
    builder_cfg = {
        'name': name,
        'avatar': bot_avatar,
        'description': description,
        'instruction': instructions,
        'conversation_starters': [row[0] for row in suggestions_filtered],
        'knowledge': list(map(lambda file: file.name, files or [])),
        'tools': {
            capability: dict(
                name=tool_cfg[capability]['name'],
                is_active=tool_cfg[capability]['is_active'],
                use=True if capability in capabilities_checkboxes else False)
            for capability in map(lambda item: item[1], capabilities)
        },
        'model': model,
    }

    save_builder_configuration(builder_cfg, uuid_str)
    update_builder(uuid_str, state)
    init_user(uuid_str, state)
    return [
        gr.HTML.update(
            visible=True,
            value=format_cover_html(builder_cfg, bot_avatar_path)),
        gr.Chatbot.update(
            visible=False,
            avatar_images=get_avatar_image(bot_avatar, uuid_str)),
        gr.Dataset.update(samples=suggestions_filtered),
        gr.DataFrame.update(value=suggestions_filtered)
    ]


# 创建 Gradio 界面
demo = gr.Blocks(css='assets/app.css')
with demo:
    uuid_str = gr.Textbox(label='modelscope_uuid', visible=False)
    draw_seed = random.randint(0, 1000000000)
    state = gr.State({'session_seed': draw_seed})
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as tabs:
                configure_tab = gr.Tab('Configure', id=1)
                with configure_tab:
                    with gr.Column():
                        # "Configure" 标签页的配置输入字段
                        with gr.Row():
                            bot_avatar_comp = gr.Image(
                                label='Avatar',
                                placeholder='Chatbot avatar image',
                                source='upload',
                                interactive=True,
                                type='filepath',
                                scale=1,
                                width=182,
                                height=182,
                            )
                            with gr.Column(scale=4):
                                name_input = gr.Textbox(
                                    label='Name', placeholder='Name your GPT')
                                description_input = gr.Textbox(
                                    label='Description',
                                    placeholder=
                                    'Add a short description about what this GPT does'
                                )

                        instructions_input = gr.Textbox(
                            label='Instructions',
                            placeholder=
                            'What does this GPT do? How does it behave? What should it avoid doing?',
                            lines=3)
                        model_selector = model_selector = gr.Dropdown(
                            label='model')
                        suggestion_input = gr.Dataframe(
                            show_label=False,
                            value=[['']],
                            datatype=['str'],
                            headers=['prompt suggestion'],
                            type='array',
                            col_count=(1, 'fixed'),
                            interactive=True)
                        knowledge_input = gr.File(
                            label='Knowledge',
                            file_count='multiple',
                            file_types=['text', '.json', '.csv'])
                        capabilities_checkboxes = gr.CheckboxGroup(
                            label='Capabilities')

                        with gr.Accordion('配置选项', open=False):
                            schema1 = gr.Textbox(
                                label='Schema',
                                placeholder='Enter your OpenAPI schema here')
                            auth1 = gr.Radio(
                                label='Authentication',
                                choices=['None', 'API Key', 'OAuth'])
                            privacy_policy1 = gr.Textbox(
                                label='Privacy Policy',
                                placeholder='Enter privacy policy URL')

                        configure_button = gr.Button('Update Configuration')

                with gr.TabItem('Create', id=0):
                    with gr.Column():
                        # "Create" 标签页的 Chatbot 组件
                        create_chatbot = gr.Chatbot(label='Create Chatbot')
                        create_chat_input = gr.Textbox(
                            label='Message',
                            placeholder='Type a message here...')
                        create_send_button = gr.Button('Send')

        with gr.Column():
            # Preview
            gr.HTML("""<div class="preview_header">Preview<div>""")

            user_chat_bot_cover = gr.HTML(format_cover_html({}, None))
            user_chatbot = ChatBot(
                value=[[None, None]],
                elem_id='user_chatbot',
                elem_classes=['markdown-body'],
                avatar_images=get_avatar_image('', uuid_str),
                height=650,
                latex_delimiters=[],
                show_label=False,
                visible=False)
            preview_chat_input = gr.Textbox(
                label='Send a message', placeholder='Type a message...')
            user_chat_bot_suggest = gr.Dataset(
                label='Prompt Suggestions',
                components=[preview_chat_input],
                samples=[])
            preview_send_button = gr.Button('Send')
            user_chat_bot_suggest.select(
                lambda evt: evt[0],
                inputs=[user_chat_bot_suggest],
                outputs=[preview_chat_input])

    configure_updated_outputs = [
        state,
        # config form
        bot_avatar_comp,
        name_input,
        description_input,
        instructions_input,
        model_selector,
        suggestion_input,
        knowledge_input,
        capabilities_checkboxes,
        # bot
        user_chat_bot_cover,
        user_chat_bot_suggest
    ]

    # tab 切换的事件处理
    def on_congifure_tab_select(_state, uuid_str):
        uuid_str = check_uuid(uuid_str)
        configure_updated = _state.get('configure_updated', False)
        if configure_updated:
            builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
                uuid_str)
            _state['configure_updated'] = False
            return init_ui_config(uuid_str, _state, builder_cfg, model_cfg,
                                  tool_cfg)
        else:
            return {state: _state}

    configure_tab.select(
        on_congifure_tab_select,
        inputs=[state, uuid_str],
        outputs=configure_updated_outputs)

    # 配置 "Create" 标签页的消息发送功能
    def format_message_with_builder_cfg(_state, chatbot, builder_cfg,
                                        uuid_str):
        uuid_str = check_uuid(uuid_str)
        bot_avatar = builder_cfg.get('avatar', '')
        conversation_starters = builder_cfg.get('conversation_starters', [])
        suggestion = [[row] for row in conversation_starters]
        bot_avatar_path = get_avatar_image(bot_avatar, uuid_str)[1]
        save_builder_configuration(builder_cfg, uuid_str)
        _state['configure_updated'] = True
        return {
            create_chatbot:
            chatbot,
            user_chat_bot_cover:
            gr.HTML.update(
                visible=True,
                value=format_cover_html(builder_cfg, bot_avatar_path)),
            user_chatbot:
            gr.Chatbot.update(
                visible=False,
                avatar_images=get_avatar_image(bot_avatar, uuid_str)),
            user_chat_bot_suggest:
            gr.Dataset.update(samples=suggestion)
        }

    def create_send_message(chatbot, input, _state, uuid_str):
        uuid_str = check_uuid(uuid_str)
        # 将发送的消息添加到聊天历史
        builder_agent = _state['builder_agent']
        chatbot.append((input, ''))
        yield {
            create_chatbot: chatbot,
            create_chat_input: gr.Textbox.update(value=''),
        }
        response = ''
        for frame in builder_agent.stream_run(input, print_info=True):
            llm_result = frame.get('llm_text', '')
            exec_result = frame.get('exec_result', '')
            print(frame)
            if len(exec_result) != 0:
                if isinstance(exec_result, dict):
                    exec_result = exec_result['result']
                    assert isinstance(exec_result, Config)
                    yield format_message_with_builder_cfg(
                        _state,
                        chatbot,
                        exec_result.to_dict(),
                        uuid_str=uuid_str)
            else:
                # llm result
                if isinstance(llm_result, dict):
                    content = llm_result['content']
                else:
                    content = llm_result
                frame_text = content
                response = f'{response}\n{frame_text}'
                chatbot[-1] = (input, response)
                yield {
                    create_chatbot: chatbot,
                }

    create_send_button.click(
        create_send_message,
        inputs=[create_chatbot, create_chat_input, state, uuid_str],
        outputs=[
            create_chatbot, user_chat_bot_cover, user_chatbot,
            user_chat_bot_suggest, create_chat_input
        ])

    # 配置 "Configure" 标签页的提交按钮功能
    configure_button.click(
        process_configuration,
        inputs=[
            uuid_str, bot_avatar_comp, name_input, description_input,
            instructions_input, model_selector, suggestion_input,
            knowledge_input, capabilities_checkboxes, state
        ],
        outputs=[
            user_chat_bot_cover, user_chatbot, user_chat_bot_suggest,
            suggestion_input
        ])

    # 配置 "Preview" 的消息发送功能
    def preview_send_message(chatbot, input, _state):
        # 将发送的消息添加到聊天历史
        user_agent = _state['user_agent']
        chatbot.append((input, ''))
        yield {
            user_chatbot: gr.Chatbot.update(visible=True, value=chatbot),
            user_chat_bot_cover: gr.HTML.update(visible=False),
            preview_chat_input: gr.Textbox.update(value='')
        }

        response = ''

        for frame in user_agent.stream_run(
                input, print_info=True, remote=False):
            # is_final = frame.get("frame_is_final")
            llm_result = frame.get('llm_text', '')
            exec_result = frame.get('exec_result', '')
            print(frame)
            # llm_result = llm_result.split("<|user|>")[0].strip()
            if len(exec_result) != 0:
                # action_exec_result
                if isinstance(exec_result, dict):
                    exec_result = str(exec_result['result'])
                frame_text = f'Observation: <result>{exec_result}</result>'
            else:
                # llm result
                frame_text = llm_result

            # important! do not change this
            response += frame_text
            chatbot[-1] = (input, response)
            yield {user_chatbot: chatbot}

    preview_send_button.click(
        preview_send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, user_chat_bot_cover, preview_chat_input])

    demo.load(
        init_all, inputs=[uuid_str, state], outputs=configure_updated_outputs)

demo.queue(concurrency_count=10)
demo.launch()
