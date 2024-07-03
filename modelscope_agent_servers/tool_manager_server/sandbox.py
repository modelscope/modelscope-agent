import time
from typing import List, Tuple

import docker
from docker.models.containers import Container
from modelscope_agent_servers.tool_manager_server.connections import \
    get_docker_client
from modelscope_agent_servers.tool_manager_server.models import \
    ToolRegisterInfo

TIMEOUT = 120
NODE_NETWORK = 'host'


def get_docker_container(tool: ToolRegisterInfo):
    docker_client = get_docker_client()
    try:
        container = docker_client.containers.get(tool.node_name)
        return container
    except docker.errors.NotFound:
        return None


def stop_docker_container(tool: ToolRegisterInfo):
    docker_client = get_docker_client()
    try:
        container = docker_client.containers.get(tool.node_name)
        if container is None:
            return
        container.stop()
        container.remove()
        elapsed = 0
        while container.status != 'exited':
            time.sleep(1)
            elapsed += 1
            if elapsed > TIMEOUT:
                break
            container = docker_client.containers.get(tool.node_name)
    except docker.errors.NotFound:
        pass


def init_docker_container(docker_client, tool: ToolRegisterInfo):
    # initialize the docker client
    container_port = f'{tool.port}/tcp'  # inside container port（protocal could be tcp/udp）
    host_port = tool.port  # host port
    port_bindings = {container_port: host_port}
    command = f'/bin/bash /app/run_tool_node.sh {tool.port}'

    container = docker_client.containers.run(
        tool.image,
        command=command,
        # network_mode=NODE_NETWORK,
        working_dir='/app',
        name=tool.node_name,
        detach=True,
        ports=port_bindings,
        environment={
            'TOOL_OSS_URL': tool.tool_url,
            'TOOL_NAME': tool.tool_name
        })
    return container


def get_exec_cmd(cmd: str) -> List[str]:
    return ['/bin/bash', '-c', cmd]


def write_tool_config(container: Container, tool: ToolRegisterInfo):
    """write tool config into docker container"""

    # serialize tool config into JSON
    tool_json = {
        'name': tool.tool_name,
        tool.tool_name: tool.config,
    }

    import json
    tool_json_str = json.dumps(tool_json)

    # get the bash command to write the JSON into a file
    cmd = f"mkdir -p /app/assets; echo '{tool_json_str}' > /app/assets/configuration.json"

    # running cmd in container
    exit_code, output = container.exec_run(
        get_exec_cmd(cmd),
        workdir=
        '/app'  # make sure the command is executed in the right directory
    )

    if exit_code != 0:
        raise Exception(
            f"Failed to write configuration file in container, exit code: {exit_code}, output: {output.decode('utf-8')}"
        )


def inject_tool_info_to_container(container: Container,
                                  tool: ToolRegisterInfo):
    if tool.node_name.startswith('http'):
        pass
    else:
        write_tool_config(container, tool)


def start_docker_container(tool: ToolRegisterInfo):
    docker_client = get_docker_client()
    try:
        container = init_docker_container(docker_client, tool)
        # wait for container to be ready
        elapsed = 0
        while container.status != 'running':
            if container.status == 'exited':
                print('container exited')
                print('container logs:')
                print(container.logs())
                break
            time.sleep(1)
            elapsed += 1
            container = docker_client.containers.get(tool.node_name)
            if elapsed > TIMEOUT:
                break
        # make configuration for class or copy remote github repo to docker container
        # inject_tool_info_to_container(container, tool)
        return container
    except Exception as e:
        raise Exception(
            f'Failed to start container for {tool.node_name}, with detail {e}')


def restart_docker_container(tool: ToolRegisterInfo):
    try:
        stop_docker_container(tool)
    except docker.errors.DockerException as e:
        print(f'Failed to stop container: {e}')
        raise e

    container = start_docker_container(tool)
    return container


def remove_docker_container(tool: ToolRegisterInfo):
    docker_client = get_docker_client()
    try:
        container = docker_client.containers.get(tool.node_name)
        if container is None:
            return
        container.remove(force=True)
    except Exception:
        raise Exception(
            f'Failed to remove container for {tool.node_name}, it might has been removed already'
        )
