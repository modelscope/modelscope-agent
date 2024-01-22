import os
import random

import json
import time
import traceback

import requests

from builder_core import beauty_output, gen_response_and_process
from config_utils import (DEFAULT_AGENT_DIR, Config, get_user_ci_dir, get_user_dir,
                          parse_configuration, save_builder_configuration)
from flask import (Flask, Response, jsonify, make_response, request,
                   send_from_directory)
from publish_util import pop_user_info_from_config, prepare_agent_zip, reload_agent_dir
from server_utils import STATIC_FOLDER, IMPORT_ZIP_TEMP_DIR, SessionManager, unzip_with_folder
from server_logging import request_id_var, logger
from modelscope_agent.schemas import Message

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')

app.session_manager = SessionManager()


@app.before_request
def set_request_id():
    request_id = request.headers.get('X-AgentFabric-Request-Id', 'unknown')
    request_id_var.set(request_id)


# builder对话接口
@app.route('/builder/chat/<uuid_str>', methods=['POST'])
def builder_chat(uuid_str):
    params_str = request.form.get('params')
    params = json.loads(params_str)
    input_content = params.get('content')
    files = request.files.getlist('files')
    file_paths = []
    for file in files:
        ci_dir = get_user_ci_dir(uuid_str)
        os.makedirs(ci_dir, exist_ok=True)
        file_path = os.path.join(ci_dir, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    def generate():
        builder_agent, builder_memory = app.session_manager.get_builder_bot(uuid_str)
        builder_memory.history = builder_memory.load_memory()

        logger.info(f'input_content: {input_content}')
        response = ''
        is_final = False

        for frame in gen_response_and_process(
                builder_agent,
                query=input_content,
                memory=builder_memory,
                print_info=True,
                uuid_str=uuid_str):

            llm_result = frame.get('llm_text', '')
            exec_result = frame.get('exec_result', '')
            step_result = frame.get('step', '')
            logger.info(f"frame, {frame}")
            if len(exec_result) != 0:
                if isinstance(exec_result, dict):
                    exec_result = exec_result['result']
                    assert isinstance(exec_result, Config)
                    builder_cfg = exec_result.to_dict()
                    save_builder_configuration(builder_cfg, uuid_str)
                    # app.session_manager.clear_user_bot(uuid_str)
                    res = json.dumps({
                        'data': response,
                        'config': builder_cfg,
                        'is_final': is_final,
                    }, ensure_ascii=False)
                    logger.info(f'res: {res}')
                    yield f'data: {res}\n\n'
            else:
                # llm result
                if isinstance(llm_result, dict):
                    content = llm_result['content']
                else:
                    content = llm_result
                if frame.get('is_final', False):
                    is_final = True
                frame_text = content
                response = beauty_output(f'{response}{frame_text}',
                                         step_result)
                res = json.dumps({
                    'data': response,
                    'is_final': is_final,
                }, ensure_ascii=False)
                logger.info(f'res: {res}')
                yield f'data: {res}\n\n'

        final_res = json.dumps({
            'data': response,
            'is_final': True,
            'request_id': request_id_var.get("")
        }, ensure_ascii=False)
        yield f'data: {final_res}\n\n'

        builder_memory.save_memory(builder_memory.history)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/builder/chat/<uuid_str>', methods=['DELETE'])
def delete_builder_chat(uuid_str):
    app.session_manager.clear_builder_bot(uuid_str)
    return jsonify({
        'success': True,
        'request_id': request_id_var.get("")
    })


@app.route('/builder/chat/<uuid_str>', methods=['GET'])
def get_builder_chat_history(uuid_str):
    _, builder_memory = app.session_manager.get_builder_bot(uuid_str)
    return jsonify({
        'history': builder_memory.get_history(),
        'success': True,
        'request_id': request_id_var.get("")
    })


# builder导入配置
@app.route('/builder/import/<uuid_str>', methods=['POST'])
def import_builder(uuid_str):
    # 检查是否有文件被上传
    if 'file' in request.files:
        # 获取上传的文件
        file = request.files['file']

        # 如果用户没有选择文件
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 保存文件到服务器的文件系统
        file_path = os.path.join(IMPORT_ZIP_TEMP_DIR, uuid_str, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        archive_dir = unzip_with_folder(file_path)
        # return jsonify({'message': 'File uploaded successfully', 'path': file_path}), 201
    else:
        url = request.get_json().get("url")
        # 发起请求获取数据
        response = requests.get(url)

        file_name = 'archive.zip'
        file_path = os.path.join(IMPORT_ZIP_TEMP_DIR, uuid_str, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        archive_dir = unzip_with_folder(file_path)
    logger.info(f"archive_dir: {archive_dir}")
    reload_agent_dir(archive_dir, DEFAULT_AGENT_DIR, uuid_str)

    return jsonify({
        'success': True,
        'request_id': request_id_var.get("")
    })


# 获取用户当前builder config
@app.route('/builder/config/<uuid_str>/')
def get_builder_config(uuid_str):
    builder_cfg, model_cfg, tool_cfg, available_tool_list, _, _ = parse_configuration(
        uuid_str)
    data = {
        'builder_config': builder_cfg.to_dict(),
        'model_config': model_cfg.to_dict(),
        'tool_config': tool_cfg.to_dict(),
        'available_tool_list': available_tool_list,
    }
    logger.info(f"preview_config: {json.dumps(data, ensure_ascii=False)}")
    return jsonify({
        'success': True,
        'data': data,
        'request_id': request_id_var.get("")
    })


# 获取用户当前builder额外文件，例如头像、上传的其他知识库等
@app.route('/builder/config_files/<uuid_str>/<file_name>', methods=['GET'])
def get_builder_file(uuid_str, file_name):
    logger.info(f'uuid_str: {uuid_str} file_name: {file_name}')
    as_attachment = request.args.get('as_attachment') == 'true'
    directory = get_user_dir(uuid_str)
    try:
        if '../' in file_name:
            raise Exception("Access not allowed.")
        response = make_response(
            send_from_directory(
                directory, file_name, as_attachment=as_attachment))
        return response
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 404,
            'message': str(e),
            'request_id': request_id_var.get("")
        }), 404


# 保存用户当前配置
@app.route('/builder/update/<uuid_str>', methods=['POST'])
def save_builder_config(uuid_str):
    builder_config_str = request.form.get('builder_config')
    builder_config = json.loads(builder_config_str)
    files = request.files.getlist('files')
    upload_dir = get_user_dir(uuid_str)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    for file in files:
        file.save(os.path.join(upload_dir, file.filename))
    save_builder_configuration(builder_cfg=builder_config, uuid_str=uuid_str)
    # app.session_manager.clear_user_bot(uuid_str)

    return jsonify({
        'success': True,
        'request_id': request_id_var.get("")
    })


# 获取用户发布包
@app.route('/builder/publish/zip/<uuid_str>', methods=['GET'])
def preview_publish_get_zip(uuid_str):
    name = f"publish_{uuid_str}"
    env_params = {}
    env_params.update(
        pop_user_info_from_config(DEFAULT_AGENT_DIR, uuid_str))
    output_url, envs_required = prepare_agent_zip(name, DEFAULT_AGENT_DIR,
                                                  uuid_str, None)
    env_params.update(envs_required)
    return jsonify({'success': True,
                    'output_url': output_url,
                    'envs_required': envs_required,
                    'request_id': request_id_var.get(""),
                    })


# 预览对话接口
@app.route('/preview/chat/<uuid_str>/<session_str>', methods=['POST'])
def preview_chat(uuid_str, session_str):
    params_str = request.form.get('params')
    params = json.loads(params_str)
    input_content = params.get('content')
    files = request.files.getlist('files')
    file_paths = []
    for file in files:
        ci_dir = get_user_ci_dir(uuid_str, session_str)
        os.makedirs(ci_dir, exist_ok=True)
        file_path = os.path.join(ci_dir, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    def generate():
        seed = random.randint(0, 1000000000)
        user_agent, user_memory = app.session_manager.get_user_bot(uuid_str, session_str)
        user_agent.seed = seed

        # get chat history from memory
        user_memory.history = user_memory.load_memory()
        history = user_memory.get_history()

        # get knowledge from memory, currently get one file
        uploaded_file = None
        if len(file_paths) > 0:
            uploaded_file = file_paths[0]
        ref_doc = user_memory.run(
            query=input_content, url=uploaded_file, checked=True)

        response = ''

        print('input_content:', input_content)
        is_final = False
        for frame in user_agent.run(
                input_content,
                history=history,
                ref_doc=ref_doc,
                append_files=file_paths,
                uuid_str=uuid_str):
            logger.info(f'frame: {frame}')
            # important! do not change this
            response += frame
            res = json.dumps({
                'data': response,
                'request_id': request_id_var.get("")
            }, ensure_ascii=False)
            yield f'data: {res}\n\n'

        if len(history) == 0:
            user_memory.update_history(
                Message(role='system', content=user_agent.system_prompt))

        user_memory.update_history([
            Message(role='user', content=input_content),
            Message(role='assistant', content=response),
        ])
        user_memory.save_memory(user_memory.history)

    return Response(generate(), mimetype='text/event-stream')


# 清除当前预览对话实例
@app.route('/preview/chat/<uuid_str>/<session_str>', methods=['DELETE'])
def delete_preview_chat(uuid_str, session_str):
    app.session_manager.clear_user_bot(uuid_str, session_str)
    return jsonify({'success': True, 'request_id': request_id_var.get("")})


@app.route('/preview/chat/<uuid_str>/<session_str>', methods=['GET'])
def get_preview_chat_history(uuid_str, session_str):
    _, user_memory = app.session_manager.get_user_bot(uuid_str, session_str)
    return jsonify({
        'history': user_memory.get_history(),
        'success': True,
        'request_id': request_id_var.get("")
    })


@app.errorhandler(Exception)
def handle_error(error):
    stack_trace = traceback.format_exc()
    stack_trace = stack_trace.replace("\n", "\\n")
    logger.error(stack_trace)
    # 处理错误并返回统一格式的错误信息
    error_message = {'success': False,
                     'message': str(error),
                     'status': 500,
                     'request_id': request_id_var.get("")
                     }
    return jsonify(error_message), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", "5001"))
    app.run(host='0.0.0.0', port=port, debug=False)
