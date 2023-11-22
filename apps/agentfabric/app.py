import random
import sys
import traceback

import gradio as gr
from builder_core import init_builder_chatbot_agent
from config_utils import (Config, get_avatar_image, parse_configuration,
                          save_avatar_image, save_builder_configuration)
from gradio_utils import ChatBot, format_cover_html
from user_core import init_user_chatbot_agent


def create_send_message(preview_chatbot, preview_chat_input, state):
    # 将发送的消息添加到聊天历史
    builder_agent = state['builder_agent']
    preview_chatbot.append((preview_chat_input, ''))
    yield format_create_send_message_ret(state, preview_chatbot)
    response = ''
    for frame in builder_agent.stream_run(preview_chat_input, print_info=True):
        llm_result = frame.get('llm_text', '')
        exec_result = frame.get('exec_result', '')
        print(frame)
        if len(exec_result) != 0:
            if isinstance(exec_result, dict):
                exec_result = exec_result['result']
                assert isinstance(exec_result, Config)
                yield format_create_send_message_ret(state, preview_chatbot,
                                                     exec_result.to_dict())
        else:
            # llm result
            if isinstance(llm_result, dict):
                content = llm_result['content']
            else:
                content = llm_result
            frame_text = content
            response = f'{response}\n{frame_text}'
            preview_chatbot[-1] = (preview_chat_input, response)
            yield format_create_send_message_ret(state, preview_chatbot)


def format_create_send_message_ret(state, chatbot, builder_cfg=None):
    if builder_cfg:
        bot_avatar = builder_cfg.get('avatar', '')
        conversation_starters = builder_cfg.get('conversation_starters', [])
        suggestion = [[row] for row in conversation_starters]
        bot_avatar_path = get_avatar_image(bot_avatar)[1]
        save_builder_configuration(builder_cfg)
        state['configure_updated'] = True
        return [
            state, chatbot,
            gr.HTML.update(
                visible=True,
                value=format_cover_html(builder_cfg, bot_avatar_path)),
            gr.Chatbot.update(
                visible=False, avatar_images=get_avatar_image(bot_avatar)),
            gr.Dataset.update(samples=suggestion)
        ]
    else:
        return [
            state, chatbot,
            gr.HTML.update(),
            gr.Chatbot.update(),
            gr.Dataset.update(samples=None)
        ]


def init_user(state):
    try:
        seed = state.get('session_seed', random.randint(0, 1000000000))
        user_agent = init_user_chatbot_agent()
        user_agent.seed = seed
        state['user_agent'] = user_agent
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    return state


def init_builder(state):
    try:
        builder_agent = init_builder_chatbot_agent()
        state['builder_agent'] = builder_agent
    except Exception as e:
        error = traceback.format_exc()
        print(f'Error:{e}, with detail: {error}')
    return state


def init_ui_config(state, builder_cfg, model_cfg, tool_cfg):
    print('builder_cfg:', builder_cfg)
    # available models
    models = list(model_cfg.keys())
    capabilities = [(tool_cfg[tool_key]['name'], tool_key)
                    for tool_key in tool_cfg.keys()
                    if tool_cfg[tool_key].get('is_active', False)]
    state['model_cfg'] = model_cfg
    state['tool_cfg'] = tool_cfg
    state['capabilities'] = capabilities
    bot_avatar = get_avatar_image(builder_cfg.get('avatar', ''))[1]
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
    return state


def init_all(state):
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
    )
    ret = init_ui_config(state, builder_cfg, model_cfg, tool_cfg)
    yield ret
    init_user(state)
    init_builder(state)
    yield ret


def format_preview_send_message_ret(preview_chatbot):
    return [
        gr.Chatbot.update(visible=True, value=preview_chatbot),
        gr.HTML.update(visible=False)
    ]


def preview_send_message(preview_chatbot, preview_chat_input, state):
    # 将发送的消息添加到聊天历史
    user_agent = state['user_agent']
    preview_chatbot.append((preview_chat_input, ''))
    yield format_preview_send_message_ret(preview_chatbot)

    response = ''

    for frame in user_agent.stream_run(
            preview_chat_input, print_info=True, remote=False):
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
        preview_chatbot[-1] = (preview_chat_input, response)
        yield format_preview_send_message_ret(preview_chatbot)


def process_configuration(bot_avatar, name, description, instructions, model,
                          suggestions, files, capabilities_checkboxes, state):
    tool_cfg = state['tool_cfg']
    capabilities = state['capabilities']

    bot_avatar, bot_avatar_path = save_avatar_image(bot_avatar)
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

    save_builder_configuration(builder_cfg)
    init_user(state)
    return [
        gr.HTML.update(
            visible=True,
            value=format_cover_html(builder_cfg, bot_avatar_path)),
        gr.Chatbot.update(
            visible=False, avatar_images=get_avatar_image(bot_avatar)),
        gr.Dataset.update(samples=suggestions_filtered),
        gr.DataFrame.update(value=suggestions_filtered)
    ]


# 创建 Gradio 界面
demo = gr.Blocks(css='assets/app.css')
with demo:
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
                avatar_images=get_avatar_image(''),
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
    def on_congifure_tab_select(_state):
        configure_updated = _state.get('configure_updated', False)
        if configure_updated:
            builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
            )
            _state['configure_updated'] = False
            return init_ui_config(_state, builder_cfg, model_cfg, tool_cfg)
        else:
            return {state: _state}

    configure_tab.select(
        on_congifure_tab_select,
        inputs=[state],
        outputs=configure_updated_outputs)

    # 配置 "Create" 标签页的消息发送功能
    create_send_button.click(
        create_send_message,
        inputs=[create_chatbot, create_chat_input, state],
        outputs=[
            state, create_chatbot, user_chat_bot_cover, user_chatbot,
            user_chat_bot_suggest
        ])

    # 配置 "Configure" 标签页的提交按钮功能
    configure_button.click(
        process_configuration,
        inputs=[
            bot_avatar_comp, name_input, description_input, instructions_input,
            model_selector, suggestion_input, knowledge_input,
            capabilities_checkboxes, state
        ],
        outputs=[
            user_chat_bot_cover, user_chatbot, user_chat_bot_suggest,
            suggestion_input
        ])

    preview_send_button.click(
        preview_send_message,
        inputs=[user_chatbot, preview_chat_input, state],
        outputs=[user_chatbot, user_chat_bot_cover])

    demo.load(init_all, inputs=[state], outputs=configure_updated_outputs)

demo.queue()
demo.launch()
