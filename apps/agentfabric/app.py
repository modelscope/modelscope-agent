import copy
import ctypes
import gc
import os
import platform
import random
import shutil
import traceback

import gradio as gr
import json
import modelscope_studio as mgr
import yaml
from builder_core import (beauty_output, gen_response_and_process,
                          init_builder_chatbot_agent)
from config_utils import (DEFAULT_AGENT_DIR, Config, get_avatar_image,
                          get_ci_dir, get_user_dir,
                          is_valid_plugin_configuration, parse_configuration,
                          save_avatar_image, save_builder_configuration,
                          save_plugin_configuration)
from gradio_utils import format_cover_html, format_goto_publish_html
from i18n import I18n
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_studio.components.Chatbot.llm_thinking_presets import qwen
from publish_util import (pop_user_info_from_config, prepare_agent_zip,
                          reload_agent_zip)
from user_core import init_user_chatbot_agent


def init_user(uuid_str, state, _user_token=None):
    try:
        allow_tool_hub = False  # modelscope-agent < 0.6.4 will be false to disable tool hub
        in_ms_studio = os.getenv('MODELSCOPE_ENVIRONMENT',
                                 'None') == 'studio' and allow_tool_hub
        seed = state.get('session_seed', random.randint(0, 1000000000))
        user_agent, user_memory = init_user_chatbot_agent(
            uuid_str, use_tool_api=in_ms_studio, user_token=_user_token)
        user_agent.seed = seed
        state['user_agent'] = user_agent
        state['user_memory'] = user_memory
    except Exception as e:
        logger.query_error(
            uuid=uuid_str,
            error=str(e),
            details={'error_traceback': traceback.format_exc()})
    return state


def init_builder(uuid_str, state):
    try:
        builder_agent, builder_memory = init_builder_chatbot_agent(uuid_str)
        state['builder_agent'] = builder_agent
        state['builder_memory'] = builder_memory
    except Exception as e:
        logger.query_error(
            uuid=uuid_str,
            error=str(e),
            details={'error_traceback': traceback.format_exc()})
    return state


def update_builder(uuid_str, state):

    try:
        builder_agent, builder_memory = init_builder_chatbot_agent(uuid_str)
        state['builder_agent'] = builder_agent
        state['builder_memory'] = builder_memory
    except Exception as e:
        logger.query_error(
            uuid=uuid_str,
            error=str(e),
            details={'error_traceback': traceback.format_exc()})

    return state


def check_uuid(uuid_str):
    if not uuid_str or uuid_str == '':
        if os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio':
            raise gr.Error('请登陆后使用! (Please login first)')
        else:
            uuid_str = 'local_user'
    return uuid_str


def delete(state):
    keys = copy.deepcopy(list(state.keys()))
    for key in keys:
        logger.info(f'Deleting the key {key}, value {state[key]}')
        del state[key]
        gc.collect()
        if platform.uname()[0] != 'Darwin':
            libc = ctypes.cdll.LoadLibrary('libc.{}'.format('so.6'))
            libc.malloc_trim(0)


# 创建 Gradio 界面
demo = gr.Blocks(css='assets/app.css')
with demo:
    user_token = gr.Textbox(label='modelscope_agent_tool_token', visible=False)
    uuid_str = gr.Textbox(label='modelscope_uuid', visible=False)
    draw_seed = random.randint(0, 1000000000)
    state = gr.State({'session_seed': draw_seed}, delete_callback=delete)
    i18n = I18n('zh-cn')
    with gr.Row():
        with gr.Column(scale=5):
            header = gr.Markdown(i18n.get('header'))
        with gr.Column(scale=1):
            language = gr.Dropdown(
                choices=[('中文', 'zh-cn'), ('English', 'en')],
                show_label=False,
                container=False,
                value='zh-cn',
                interactive=True)
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as tabs:
                with gr.Tab(i18n.get_whole('create'), id=0) as create_tab:
                    with gr.Column():
                        # "Create" 标签页的 Chatbot 组件
                        start_text = '欢迎使用agent创建助手。我可以帮助您创建一个定制agent。'\
                            '您希望您的agent主要用于什么领域或任务？比如，您可以说，我想做一个RPG游戏agent'
                        create_chatbot = mgr.Chatbot(
                            show_label=False,
                            value=[[None, start_text]],
                            flushing=False,
                            show_copy_button=True,
                            llm_thinking_presets=[
                                qwen(
                                    action_input_title='调用 <Action>',
                                    action_output_title='完成调用')
                            ])
                        create_chat_input = mgr.MultimodalInput(
                            label=i18n.get('message'),
                            placeholder=i18n.get('message_placeholder'),
                            interactive=False,
                            upload_button_props=dict(visible=False),
                            submit_button_props=dict(
                                label=i18n.get('sendOnLoading')))

                configure_tab = gr.Tab(i18n.get_whole('configure'), id=1)
                with configure_tab:
                    with gr.Column():
                        # "Configure" 标签页的配置输入字段
                        with gr.Row():
                            bot_avatar_comp = gr.Image(
                                label=i18n.get('form_avatar'),
                                sources=['upload'],
                                interactive=True,
                                type='filepath',
                                scale=1,
                                width=182,
                                height=182,
                            )
                            with gr.Column(scale=4):
                                name_input = gr.Textbox(
                                    label=i18n.get('form_name'),
                                    placeholder=i18n.get(
                                        'form_name_placeholder'))
                                description_input = gr.Textbox(
                                    label=i18n.get('form_description'),
                                    placeholder=i18n.get(
                                        'form_description_placeholder'))

                        instructions_input = gr.Textbox(
                            label=i18n.get('form_instructions'),
                            placeholder=i18n.get(
                                'form_instructions_placeholder'),
                            lines=3)
                        model_selector = gr.Dropdown(
                            label=i18n.get('form_model'))
                        agent_language_selector = gr.Dropdown(
                            label=i18n.get('form_agent_language'),
                            choices=['zh', 'en'],
                            value='zh')
                        prologue_input = gr.Textbox(
                            label=i18n.get('form_prologue'),
                            placeholder=i18n.get('form_prologue_placeholder'),
                            lines=2)
                        suggestion_input = gr.Dataframe(
                            show_label=False,
                            value=[['']],
                            datatype=['str'],
                            headers=[i18n.get_whole('form_prompt_suggestion')],
                            type='array',
                            col_count=(1, 'fixed'),
                            interactive=True)
                        gr.Markdown(
                            '*注意：知识库上传的文本文档默认按照\\n\\n切分，pdf默认按照页切分。如果片段'
                            '对应的字符大于[配置文件](https://github.com/modelscope/modelscope-agent/'
                            'blob/master/apps/agentfabric/config/model_config.json)中指定模型的'
                            'knowledge限制，则在被召回时有可能会被截断。*')
                        knowledge_input = gr.File(
                            label=i18n.get('form_knowledge'),
                            file_count='multiple',
                            file_types=[
                                'text', '.json', '.csv', '.pdf', '.md'
                            ])
                        knowledge_upload_button = gr.UploadButton(
                            i18n.get('form_knowledge_upload_button'),
                            file_count='multiple',
                            file_types=[
                                'text', '.json', '.csv', '.pdf', '.md'
                            ])
                        capabilities_checkboxes = gr.CheckboxGroup(
                            label=i18n.get('form_capabilities'))

                        with gr.Accordion(
                                i18n.get('open_api_accordion'),
                                open=False) as open_api_accordion:
                            openapi_schema = gr.Textbox(
                                label='Schema',
                                placeholder=
                                'Enter your OpenAPI schema here, JSON or YAML format only'
                            )

                            with gr.Group():
                                openapi_auth_type = gr.Radio(
                                    label='Authentication Type',
                                    choices=['None', 'API Key'],
                                    value='None')
                                openapi_auth_apikey = gr.Textbox(
                                    label='API Key',
                                    placeholder='Enter your API Key here')
                                openapi_auth_apikey_type = gr.Radio(
                                    label='API Key type', choices=['Bearer'])
                            openapi_privacy_policy = gr.Textbox(
                                label='Privacy Policy',
                                placeholder='Enter privacy policy URL')

                        configure_button = gr.Button(
                            i18n.get('form_update_button'))

                        with gr.Accordion(
                                label=i18n.get('import_config'),
                                open=False) as update_accordion:
                            with gr.Column():
                                update_space = gr.Textbox(
                                    label=i18n.get('space_addr'),
                                    placeholder=i18n.get('input_space_addr'))
                                import_button = gr.Button(
                                    i18n.get_whole('import_space'))
                                gr.Markdown(
                                    f'#### {i18n.get_whole("import_hint")}')

        with gr.Column():
            # Preview
            preview_header = gr.HTML(
                f"""<div class="preview_header">{i18n.get('preview')}<div>""")

            user_chat_bot_cover = gr.HTML(format_cover_html({}, None))
            user_chatbot = mgr.Chatbot(
                value=[[None, None]],
                elem_id='user_chatbot',
                elem_classes=['markdown-body'],
                avatar_images=get_avatar_image('', uuid_str),
                height=650,
                show_label=False,
                visible=False,
                show_copy_button=True,
                llm_thinking_presets=[
                    qwen(
                        action_input_title='调用 <Action>',
                        action_output_title='完成调用')
                ])
            preview_chat_input = mgr.MultimodalInput(
                interactive=False,
                label=i18n.get('message'),
                placeholder=i18n.get('message_placeholder'),
                upload_button_props=dict(
                    label=i18n.get('upload_btn'),
                    file_types=['file', 'image', 'audio', 'video', 'text'],
                    file_count='multiple'),
                submit_button_props=dict(label=i18n.get('sendOnLoading')))
            user_chat_bot_suggest = gr.Dataset(
                label=i18n.get('prompt_suggestion'),
                components=[preview_chat_input],
                samples=[])
            user_chat_bot_suggest.select(
                lambda evt: evt[0],
                inputs=[user_chat_bot_suggest],
                outputs=[preview_chat_input])
            with gr.Accordion(
                    label=i18n.get('publish'),
                    open=False) as publish_accordion:
                publish_alert_md = gr.Markdown(f'{i18n.get("publish_alert")}')
                with gr.Row():
                    with gr.Column():
                        publish_button = gr.Button(i18n.get_whole('build'))
                        build_hint_md = gr.Markdown(
                            f'#### 1.{i18n.get("build_hint")}')

                    with gr.Column():
                        publish_link = gr.HTML(
                            value=format_goto_publish_html(
                                i18n.get_whole('publish'), '', {}, True))
                        publish_hint_md = gr.Markdown(
                            f'#### 2.{i18n.get("publish_hint")}')

    configure_updated_outputs = [
        state,
        # config form
        bot_avatar_comp,
        name_input,
        description_input,
        instructions_input,
        model_selector,
        agent_language_selector,
        prologue_input,
        suggestion_input,
        knowledge_input,
        capabilities_checkboxes,
        # bot
        user_chat_bot_cover,
        user_chat_bot_suggest,
        preview_chat_input,
        create_chat_input,
    ]

    # 初始化表单
    def init_ui_config(uuid_str, _state, builder_cfg, model_cfg, tool_cfg):
        logger.query_info(
            uuid=uuid_str,
            message='builder_cfg',
            details={'builder_cfg': str(builder_cfg)})
        # available models
        models = list(model_cfg.keys())
        capabilities = [(tool_cfg[tool_key]['name'], tool_key)
                        for tool_key in tool_cfg.keys()
                        if tool_cfg[tool_key].get('is_active', False)]
        _state['model_cfg'] = model_cfg
        _state['tool_cfg'] = tool_cfg
        _state['capabilities'] = capabilities
        bot_avatar = get_avatar_image(builder_cfg.get('avatar', ''),
                                      uuid_str)[1]
        suggests = builder_cfg.get('prompt_recommend', [''])
        return {
            state:
            _state,
            bot_avatar_comp:
            gr.Image(value=bot_avatar),
            name_input:
            builder_cfg.get('name', ''),
            description_input:
            builder_cfg.get('description'),
            instructions_input:
            builder_cfg.get('instruction'),
            model_selector:
            gr.Dropdown(
                value=builder_cfg.get('model', models[0]), choices=models),
            agent_language_selector:
            builder_cfg.get('language') or 'zh',
            prologue_input:
            builder_cfg.get('prologue') or '',
            suggestion_input:
            [[str] for str in suggests] if len(suggests) > 0 else [['']],
            knowledge_input:
            builder_cfg.get('knowledge', [])
            if len(builder_cfg['knowledge']) > 0 else None,
            capabilities_checkboxes:
            gr.CheckboxGroup(
                value=[
                    tool for tool in builder_cfg.get('tools', {}).keys()
                    if builder_cfg.get('tools').get(tool).get('use', False)
                ],
                choices=capabilities),
            # bot
            user_chat_bot_cover:
            format_cover_html(builder_cfg, bot_avatar),
            user_chat_bot_suggest:
            gr.Dataset(
                components=[preview_chat_input],
                samples=[[item] for item in suggests]),
        }

    # tab 切换的事件处理
    def on_congifure_tab_select(_state, uuid_str):
        uuid_str = check_uuid(uuid_str)
        configure_updated = _state.get('configure_updated', False)
        if configure_updated:
            builder_cfg, model_cfg, tool_cfg, available_tool_list, _, _ = parse_configuration(
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

    # 上传文件到知识库的按钮
    def on_append_file_to_knowledge(files, knowledge_input):
        file_paths = [file.name for file in files]
        return (knowledge_input or []) + file_paths

    knowledge_upload_button.upload(
        on_append_file_to_knowledge,
        inputs=[knowledge_upload_button, knowledge_input],
        outputs=[knowledge_input])

    # 配置 "Create" 标签页的消息发送功能
    def format_message_with_builder_cfg(_state, chatbot, builder_cfg,
                                        uuid_str):
        uuid_str = check_uuid(uuid_str)
        bot_avatar = builder_cfg.get('avatar', '')
        prompt_recommend = builder_cfg.get('prompt_recommend', [''])
        suggestion = [[row] for row in prompt_recommend]
        bot_avatar_path = get_avatar_image(bot_avatar, uuid_str)[1]
        save_builder_configuration(builder_cfg, uuid_str)
        update_builder(uuid_str, _state)
        init_user(uuid_str, _state)
        _state['configure_updated'] = True
        return {
            create_chatbot:
            chatbot,
            user_chat_bot_cover:
            gr.HTML(
                visible=True,
                value=format_cover_html(builder_cfg, bot_avatar_path)),
            user_chatbot:
            gr.update(
                visible=False,
                avatar_images=get_avatar_image(bot_avatar, uuid_str)),
            user_chat_bot_suggest:
            gr.Dataset(components=[preview_chat_input], samples=suggestion)
        }

    def create_send_message(chatbot, input, _state, uuid_str):
        uuid_str = check_uuid(uuid_str)
        # 将发送的消息添加到聊天历史
        builder_agent = _state['builder_agent']
        builder_memory = _state['builder_memory']

        chatbot.append([{'text': input.text, 'files': input.files}, None])
        yield {
            create_chatbot: chatbot,
            create_chat_input: None,
        }
        response = ''
        for frame in gen_response_and_process(
                builder_agent,
                query=input.text,
                memory=builder_memory,
                print_info=True,
                uuid_str=uuid_str):
            llm_result = frame.get('llm_text', '')
            exec_result = frame.get('exec_result', '')
            step_result = frame.get('step', '')
            logger.query_info(
                uuid=uuid_str, message='frame', details={'frame': str(frame)})
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
                    content = llm_result.get('content', None)
                    if not content:
                        content = llm_result.get(
                            'error',
                            'llm result is invalid, please reset and try again.'
                        )
                else:
                    content = llm_result
                frame_text = content
                response = beauty_output(f'{response}{frame_text}',
                                         step_result)
                chatbot[-1][1] = response
                yield {
                    create_chatbot: chatbot,
                }

    create_chat_input.submit(
        create_send_message,
        inputs=[create_chatbot, create_chat_input, state, uuid_str],
        outputs=[
            create_chatbot, user_chat_bot_cover, user_chatbot,
            user_chat_bot_suggest, create_chat_input
        ])

    def process_configuration(uuid_str, bot_avatar, name, description,
                              instructions, model, agent_language, prologue,
                              suggestions, knowledge_files,
                              capabilities_checkboxes, openapi_schema,
                              openapi_auth, openapi_auth_apikey,
                              openapi_auth_apikey_type, openapi_privacy_policy,
                              state):
        uuid_str = check_uuid(uuid_str)
        tool_cfg = state['tool_cfg']
        capabilities = state['capabilities']
        bot_avatar, bot_avatar_path = save_avatar_image(bot_avatar, uuid_str)
        suggestions_filtered = [row for row in suggestions if row[0]]
        if len(suggestions_filtered) == 0:
            suggestions_filtered = [['']]
        user_dir = get_user_dir(uuid_str)
        if knowledge_files is not None:
            new_knowledge_files = [
                os.path.join(user_dir, os.path.basename((f.name)))
                for f in knowledge_files
            ]
            for src_file, dst_file in zip(knowledge_files,
                                          new_knowledge_files):
                if not os.path.exists(dst_file):
                    shutil.copy(src_file.name, dst_file)
        else:
            new_knowledge_files = []
        builder_cfg = {
            'name': name,
            'avatar': bot_avatar,
            'description': description,
            'instruction': instructions,
            'prologue': prologue,
            'prompt_recommend': [row[0] for row in suggestions_filtered],
            'knowledge': new_knowledge_files,
            'tools': {
                capability:
                dict([(key, value if key != 'use' else
                       (capability in capabilities_checkboxes))
                      for key, value in tool_cfg[capability].items()])
                for capability in map(lambda item: item[1], capabilities)
            },
            'model': model,
            'language': agent_language,
        }

        try:
            try:
                schema_dict = json.loads(openapi_schema)
            except json.decoder.JSONDecodeError:
                schema_dict = yaml.safe_load(openapi_schema)
            except Exception as e:
                raise gr.Error(
                    f'OpenAPI schema format error, should be one of json and yaml: {e}'
                )

            openapi_plugin_cfg = {
                'schema': schema_dict,
                'auth': {
                    'type': openapi_auth,
                    'apikey': openapi_auth_apikey,
                    'apikey_type': openapi_auth_apikey_type
                },
                'privacy_policy': openapi_privacy_policy
            }
            if is_valid_plugin_configuration(openapi_plugin_cfg):
                save_plugin_configuration(openapi_plugin_cfg, uuid_str)
        except Exception as e:
            logger.query_error(
                uuid=uuid_str,
                error=str(e),
                details={'error_traceback': traceback.format_exc()})

        save_builder_configuration(builder_cfg, uuid_str)
        update_builder(uuid_str, state)
        init_user(uuid_str, state)
        return {
            user_chat_bot_cover:
            gr.HTML(
                visible=True,
                value=format_cover_html(builder_cfg, bot_avatar_path)),
            user_chatbot:
            gr.update(
                visible=False,
                avatar_images=get_avatar_image(bot_avatar, uuid_str)),
            suggestion_input: [item[:] for item in suggestions_filtered],
            user_chat_bot_suggest:
            gr.Dataset(
                components=[preview_chat_input], samples=suggestions_filtered),
        }

    # 配置 "Configure" 标签页的提交按钮功能
    configure_button.click(
        process_configuration,
        inputs=[
            uuid_str, bot_avatar_comp, name_input, description_input,
            instructions_input, model_selector, agent_language_selector,
            prologue_input, suggestion_input, knowledge_input,
            capabilities_checkboxes, openapi_schema, openapi_auth_type,
            openapi_auth_apikey, openapi_auth_apikey_type,
            openapi_privacy_policy, state
        ],
        outputs=[
            user_chat_bot_cover, user_chatbot, user_chat_bot_suggest,
            suggestion_input
        ])

    # 配置 "Preview" 的消息发送功能
    def preview_send_message(chatbot, input, _state, uuid_str, _user_token):
        # 将发送的消息添加到聊天历史
        # _uuid_str = check_uuid(uuid_str)
        user_agent = _state['user_agent']
        append_files = []
        for file in input.files:
            file_name = os.path.basename(file.path)
            # covert xxx.json to xxx_uuid_str.json
            file_name = file_name.replace('.', f'_{uuid_str}.')
            file_path = os.path.join(get_ci_dir(), file_name)
            if not os.path.exists(file_path):
                # make sure file path's directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy(file.path, file_path)
            append_files.append(file_path)
        user_memory = _state['user_memory']

        chatbot.append([{'text': input.text, 'files': input.files}, None])
        yield {
            user_chatbot: mgr.Chatbot(visible=True, value=chatbot),
            user_chat_bot_cover: gr.HTML(visible=False),
            preview_chat_input: None
        }

        # get chat history from memory
        history = user_memory.get_history()

        use_llm = True if len(user_agent.function_list) else False
        ref_doc = user_memory.run(
            query=input.text,
            url=append_files,
            max_token=4000,
            top_k=2,
            checked=True,
            use_llm=use_llm)

        response = ''
        try:
            for frame in user_agent.run(
                    input.text,
                    history=history,
                    ref_doc=ref_doc,
                    append_files=append_files,
                    uuid_str=uuid_str,
                    user_token=_user_token):
                # append_files=new_file_paths):
                # important! do not change this
                response += frame
                chatbot[-1][1] = response
                yield {user_chatbot: chatbot}
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

    preview_chat_input.submit(
        preview_send_message,
        inputs=[user_chatbot, preview_chat_input, state, uuid_str, user_token],
        outputs=[user_chatbot, user_chat_bot_cover, preview_chat_input])

    # configuration for publish
    def publish_agent(name, uuid_str, state):
        uuid_str = check_uuid(uuid_str)
        env_params = {}
        env_params.update(
            pop_user_info_from_config(DEFAULT_AGENT_DIR, uuid_str))
        output_url, envs_required = prepare_agent_zip(name, DEFAULT_AGENT_DIR,
                                                      uuid_str, state)
        env_params.update(envs_required)
        # output_url = "https://test.url"
        return format_goto_publish_html(
            i18n.get_whole('publish'), output_url, env_params)

    publish_button.click(
        publish_agent,
        inputs=[name_input, uuid_str, state],
        outputs=[publish_link],
    )

    def import_space(agent_url, uuid_str, state):
        uuid_str = check_uuid(uuid_str)
        _ = reload_agent_zip(agent_url, DEFAULT_AGENT_DIR, uuid_str, state)

        # update config
        builder_cfg, model_cfg, tool_cfg, available_tool_list, _, _ = parse_configuration(
            uuid_str)
        return init_ui_config(uuid_str, state, builder_cfg, model_cfg,
                              tool_cfg)

    import_button.click(
        import_space,
        inputs=[update_space, uuid_str, state],
        outputs=configure_updated_outputs,
    )

    def change_lang(language):
        i18n = I18n(language)
        return {
            bot_avatar_comp:
            gr.Image(label=i18n.get('form_avatar')),
            name_input:
            gr.Textbox(
                label=i18n.get('form_name'),
                placeholder=i18n.get('form_name_placeholder')),
            description_input:
            gr.Textbox(
                label=i18n.get('form_description'),
                placeholder=i18n.get('form_description_placeholder')),
            instructions_input:
            gr.Textbox(
                label=i18n.get('form_instructions'),
                placeholder=i18n.get('form_instructions_placeholder')),
            model_selector:
            gr.Dropdown(label=i18n.get('form_model')),
            agent_language_selector:
            gr.Dropdown(label=i18n.get('form_agent_language')),
            prologue_input:
            gr.update(label=i18n.get('form_prologue')),
            knowledge_input:
            gr.File(label=i18n.get('form_knowledge')),
            knowledge_upload_button:
            gr.update(label=i18n.get('form_knowledge_upload_button')),
            capabilities_checkboxes:
            gr.CheckboxGroup(label=i18n.get('form_capabilities')),
            open_api_accordion:
            gr.Accordion(label=i18n.get('open_api_accordion')),
            configure_button:
            gr.Button(i18n.get('form_update_button')),
            preview_header:
            gr.HTML(
                f"""<div class="preview_header">{i18n.get('preview')}<div>"""),
            preview_chat_input:
            mgr.MultimodalInput(
                label=i18n.get('message'),
                placeholder=i18n.get('message_placeholder'),
                submit_button_props=dict(label=i18n.get('send')),
                upload_button_props=dict(
                    label=i18n.get('upload_btn'),
                    file_types=['file', 'image', 'audio', 'video', 'text'],
                    file_count='multiple')),
            create_chat_input:
            mgr.MultimodalInput(
                label=i18n.get('message'),
                placeholder=i18n.get('message_placeholder'),
                submit_button_props=dict(label=i18n.get('send'))),
            user_chat_bot_suggest:
            gr.Dataset(
                components=[preview_chat_input],
                label=i18n.get('prompt_suggestion')),
            publish_accordion:
            gr.Accordion(label=i18n.get('publish')),
            header:
            gr.Markdown(i18n.get('header')),
            publish_alert_md:
            gr.Markdown(f'{i18n.get("publish_alert")}'),
            build_hint_md:
            gr.Markdown(f'#### 1.{i18n.get("build_hint")}'),
            publish_hint_md:
            gr.Markdown(f'#### 2.{i18n.get("publish_hint")}'),
        }

    language.select(
        change_lang,
        inputs=[language],
        outputs=configure_updated_outputs + [
            configure_button, create_chat_input, open_api_accordion,
            preview_header, preview_chat_input, publish_accordion, header,
            publish_alert_md, build_hint_md, publish_hint_md,
            knowledge_upload_button
        ])

    def init_all(uuid_str, _state, _user_token):
        uuid_str = check_uuid(uuid_str)
        builder_cfg, model_cfg, tool_cfg, available_tool_list, _, _ = parse_configuration(
            uuid_str)
        ret = init_ui_config(uuid_str, _state, builder_cfg, model_cfg,
                             tool_cfg)
        yield ret
        init_user(uuid_str, _state, _user_token)
        init_builder(uuid_str, _state)
        yield {
            state:
            _state,
            preview_chat_input:
            mgr.MultimodalInput(
                submit_button_props=dict(label=i18n.get('send')),
                interactive=True),
            create_chat_input:
            mgr.MultimodalInput(
                submit_button_props=dict(label=i18n.get('send')),
                interactive=True),
        }

    demo.load(
        init_all,
        inputs=[uuid_str, state, user_token],
        outputs=configure_updated_outputs)

demo.queue()
demo.launch(show_error=True, max_threads=10)
