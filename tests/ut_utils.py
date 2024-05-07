import subprocess

from modelscope_agent.tools.base import BaseTool


def is_docker_daemon_running():
    try:
        # Run the 'docker info' command to check if Docker daemon is running
        result = subprocess.run(['docker', 'info'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True)
        print(result.stdout.decode('utf-8'))
        # If 'docker info' command runs successfully, the daemon is running
        return True
    except subprocess.CalledProcessError:
        # The 'docker info' command failed, so the daemon is not running or not reachable
        return False
    except FileNotFoundError:
        # The 'docker' command is not available in the system's path, so Docker is likely not installed
        return False


class MockTool(BaseTool):
    name: str = 'mock_tool'
    description: str = 'description'
    parameters: list = [{
        'name': 'test',
        'type': 'string',
        'description': 'test variable',
        'required': False
    }]

    def call(self, params: str, **kwargs):
        return params
