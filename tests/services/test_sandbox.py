import os
import time

import pytest
from modelscope_agent_servers.tool_manager_server.models import \
    ToolRegisterInfo
from modelscope_agent_servers.tool_manager_server.sandbox import (
    get_docker_container, get_exec_cmd, restart_docker_container,
    start_docker_container)

IN_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

USE_REAL_DOCKER = os.environ.get('USE_REAL_DOCKER', 'True').lower() == 'true'

if not os.path.exists('/tmp/test-tool-node'):
    os.mkdir('/tmp/test-tool-node')


@pytest.fixture
def mock_tool_info():
    return ToolRegisterInfo(
        tool_name='test_tool_name',
        node_name='test-tool',
        image='modelscope-agent/tool-node:latest',
        config={'config_key': 'config_value'},
        workspace_dir='/tmp/test-tool-node',
        tenant_id='test-tenant')


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason='Need to set up the docker environment')
def test_start_docker_container(mock_tool_info):
    container = get_docker_container(mock_tool_info)
    if container is not None:
        container.remove(force=True)

    start_time = time.time()
    if not USE_REAL_DOCKER:
        return
    container = start_docker_container(mock_tool_info)
    print(
        f'time consumed after starting container is {time.time() - start_time}'
    )
    assert container.status == 'running'

    # get the bash command to write the JSON into a file
    cmd = 'cat /app/assets/configuration.json'

    # running cmd in container
    exit_code, output = container.exec_run(
        get_exec_cmd(cmd),
        workdir=
        '/app'  # make sure the command is executed in the right directory
    )
    assert exit_code == 0
    assert output.decode(
        'utf-8'
    ) == '{"name": "test_tool_name", "test_tool_name": {"config_key": "config_value"}}\n'

    container.stop()
    container.remove(force=True)
    print(
        f'time consumed after closing container is { time.time() - start_time}'
    )


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason='Need to set up the docker environment')
def test_restart_docker_container(mock_tool_info):

    # running the origin container
    container = get_docker_container(mock_tool_info)
    if container is not None:
        container.remove(force=True)
    container = start_docker_container(mock_tool_info)
    assert container.status == 'running'

    # update the config
    mock_tool_info.config = {'config_key1': 'config_value1'}
    start_time = time.time()
    if not USE_REAL_DOCKER:
        return

    # restart the container
    container = restart_docker_container(mock_tool_info)
    print(
        f'time consumed after starting container is {time.time() - start_time}'
    )
    assert container.status == 'running'

    # get the bash command to write the JSON into a file
    cmd = 'cat /app/assets/configuration.json'

    # running cmd in container
    exit_code, output = container.exec_run(
        get_exec_cmd(cmd),
        workdir=
        '/app'  # make sure the command is executed in the right directory
    )
    assert exit_code == 0
    assert output.decode(
        'utf-8'
    ) == '{"name": "test_tool_name", "test_tool_name": {"config_key1": "config_value1"}}\n'

    container.stop()
    container.remove(force=True)
    print(
        f'time consumed after closing container is { time.time() - start_time}'
    )
