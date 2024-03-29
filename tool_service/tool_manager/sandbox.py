import concurrent.futures
import time
from typing import List, Tuple

import docker
from connections import ToolRegistration, get_docker_client

TIMEOUT = 120
NODE_NETWORK = 'tool-server-network'
NODE_PORT = 31513


def stop_docker_container(tool: ToolRegistration):
    docker_client = get_docker_client()
    try:
        container = docker_client.containers.get(tool.name)
        container.stop()
        container.remove()
        elapsed = 0
        while container.status != 'exited':
            time.sleep(1)
            elapsed += 1
            if elapsed > TIMEOUT:
                break
            container = docker_client.containers.get(tool.name)
    except docker.errors.NotFound:
        pass


def init_docker_container(docker_client, tool: ToolRegistration):
    # initialize the docker client
    container = docker_client.containers.run(
        tool.image,
        command='tail -f /dev/null',
        network_mode=NODE_NETWORK,
        working_dir='/app',
        name=tool.name,
        detach=True,
        volumes={tool.workspace_dir: {
            'bind': '/app',
            'mode': 'rw'
        }},
    )
    return container


def start_docker_container(tool: ToolRegistration):
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
            container = docker_client.containers.get(tool.name)
            if elapsed > TIMEOUT:
                break
        return container
    except Exception as e:
        raise Exception(
            f'Failed to start container for {tool.name}, with detail {e}')


def restart_docker_container(tool: ToolRegistration):
    try:
        stop_docker_container(tool)
    except docker.errors.DockerException as e:
        print(f'Failed to stop container: {e}')
        raise e

    container = start_docker_container(tool)
    return container


def remove_docker_container(tool: ToolRegistration):
    docker_client = get_docker_client()
    container = docker_client.containers.get(tool.name)
    container.remove(force=True)
