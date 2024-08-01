import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from modelscope_agent.environment.graph_database import GraphDatabaseHandler
from modelscope_agent.environment.graph_database.ast_search import AstManager


def get_py_files(directory):
    py_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files


def run_single(path, root, task_id, shallow, env_path_dict=None):

    env_path = env_path_dict['env_path']
    script_path = os.path.join(env_path_dict['working_directory'],
                               'run_index_single.py')
    working_directory = env_path_dict['working_directory']
    url = env_path_dict['url']
    user = env_path_dict['user']
    password = env_path_dict['password']
    db_name = env_path_dict['db_name']

    if shallow:
        script_args = [
            '--file_path',
            path,
            '--root_path',
            root,
            '--task_id',
            task_id,
            '--url',
            url,
            '--user',
            user,
            '--password',
            password,
            '--db_name',
            db_name,
            '--env',
            env_path,
            '--shallow',
        ]
    else:
        script_args = [
            '--file_path', path, '--root_path', root, '--task_id', task_id
        ]
    return run_script_in_env(env_path, script_path, working_directory,
                             script_args)


def run_script_in_env(env_path,
                      script_path,
                      working_directory,
                      script_args=None):
    # python_executable = os.path.join(env_path, "bin", "python")
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            'Python executable not found in the environment: {}'.format(
                env_path))

    command = [env_path, script_path]
    if script_args:
        command.extend(script_args)
    # print(' '.join(command))

    try:
        result = subprocess.run(
            command,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = result.stdout.decode('utf-8')
        stderr = result.stderr.decode('utf-8')

        if result.returncode == 0:
            return 'Script executed successfully:\n{}'.format(stdout)
        else:
            return 'Script execution failed:\n{}'.format(stderr)
    except subprocess.CalledProcessError as e:
        return 'Error: {}'.format(e.stderr)


def build_graph_database(graph_db: GraphDatabaseHandler,
                         repo_path: str,
                         task_id: str,
                         is_clear: bool = True,
                         max_workers=None,
                         env_path_dict=None,
                         update_progress_bar=None):
    file_list = get_py_files(repo_path)
    root_path = repo_path

    if is_clear:
        graph_db.clear_task_data(task_id=task_id)

    start_time = time.time()

    total_files = len(file_list)

    if update_progress_bar:
        update_progress_bar(0.5 / total_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_single, file_path, root_path, task_id, True,
                            env_path_dict): file_path
            for file_path in file_list
        }
        for i, future in enumerate(as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                future.result()
                print('Successfully processed {}'.format(file_path))
            except Exception as exc:
                msg = '`{}` generated an exception: `{}`'.format(
                    file_path, exc)
                print(msg)
                # 在捕获到异常后，停止提交新任务，并尝试取消所有未完成的任务
                executor.shutdown(wait=False, cancel_futures=True)
                return msg
            finally:
                # 每完成一个任务，更新进度条
                if update_progress_bar:
                    update_progress_bar((i + 1) / total_files)
                # print((i+1) / total_files)
    # ast, class inheritance
    ast_manage = AstManager(repo_path, task_id, graph_db)
    ast_manage.run()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'✍️ Shallow indexing ({int(elapsed_time)} s)')
    # logger.info(f"✍️ Shallow indexing ({int(elapsed_time)} s)")
    return None
