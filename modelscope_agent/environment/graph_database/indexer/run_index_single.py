import argparse
import os
import sys
import uuid

import my_client as my_client
import shallow_indexer
from my_graph_db import GraphDatabaseHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
# 确保依赖项在系统的 PATH 环境变量中
os.environ['PATH'] = os.path.dirname(
    os.path.abspath(__file__)) + ';' + os.environ['PATH']


def indexSourceFile(
    sourceFilePath,
    environmentPath,
    workingDirectory,
    graph_db: GraphDatabaseHandler,
    rootPath,
    shallow,
):
    astVisitorClient = my_client.AstVisitorClient(
        graph_db, task_root_path=rootPath)

    print('use shallow: ' + str(shallow))

    if not shallow:
        return
    else:
        shallow_indexer.indexSourceFile(
            sourceFilePath,
            environmentPath,
            workingDirectory,
            astVisitorClient,
            False,
            rootPath,
        )


def run_single(
    graph_db: GraphDatabaseHandler,
    environmentPath='',
    sourceFilePath='',
    root_path='',
    srctrl_clear=False,
    shallow=True,
):
    workingDirectory = os.getcwd()
    indexSourceFile(
        sourceFilePath,
        environmentPath,
        workingDirectory,
        graph_db,
        root_path,
        shallow,
    )


def run():
    # task_id = 'test_sh'
    parser = argparse.ArgumentParser(
        description=
        'Python source code indexer that generates a Sourcetrail compatible database.'
    )
    parser.add_argument(
        '--file_path',
        help='path to the source file to index',
        default='',
        type=str,
        required=False,
    )
    parser.add_argument('--root_path', default='', required=False)
    parser.add_argument(
        '--task_id', help='task_id', type=str, default='', required=False)

    parser.add_argument(
        '--url', help='neo4j url', type=str, default='', required=False)
    parser.add_argument(
        '--user', help='neo4j user', type=str, default='', required=False)
    parser.add_argument(
        '--password',
        help='neo4j password',
        type=str,
        default='',
        required=False)
    parser.add_argument(
        '--db_name',
        help='neo4j db name',
        type=str,
        default='',
        required=False)
    parser.add_argument(
        '--env', help='env', type=str, default='', required=False)

    parser.add_argument(
        '--shallow', help='shallow', action='store_true', required=False)

    parser.add_argument(
        '--clear', help='clear', action='store_true', required=False)

    args = parser.parse_args()

    print('start')

    task_id = args.task_id
    file_path = args.file_path
    root_path = args.root_path
    is_shallow = args.shallow
    is_clear = args.clear
    uri = args.url
    user = args.user
    password = args.password
    db_name = args.db_name
    env_path = args.env

    graph_db = GraphDatabaseHandler(
        uri=uri,
        user=user,
        password=password,
        database_name=db_name,
        task_id=task_id,
        use_lock=True,
    )
    if is_clear:
        graph_db.clear_task_data(task_id)

    run_single(
        graph_db,
        environmentPath=env_path,
        sourceFilePath=file_path,
        root_path=root_path,
        shallow=is_shallow)
    print('Success build graph')


if __name__ == '__main__':
    run()
