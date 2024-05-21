import os

import pytest
import requests
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.constants import DEFAULT_TOOL_MANAGER_SERVICE_URL

IN_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'


def check_url(url: str):
    try:
        response = requests.get(url, timeout=5)  # request url
        if response.status_code == 200:
            print(
                f'{url} is accessible and returned a successful status code.')
            return True
    except requests.ConnectionError:
        print(f'{url} is not accessible due to a connection error.')
    except requests.Timeout:
        print(f'Request to {url} timed out.')
    except requests.RequestException as e:
        print(f'An error occurred while trying to access {url}: {e}')

    return False


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason='Need to set up the docker environment')
def test_role_play_with():
    llm_config = {'model': 'qwen-turbo', 'model_server': 'dashscope'}

    # input tool name
    function_list = ['image_gen']

    is_accessible = check_url(DEFAULT_TOOL_MANAGER_SERVICE_URL)

    if not is_accessible:
        assert False, """Start up the tool manager service by `sh scripts/run_tool_manager.sh`"""

    bot = RolePlay(
        function_list=function_list, llm=llm_config, use_tool_api=True)

    response = bot.run(
        '创建一个多啦A梦', dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'))

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
    assert 'Answer:' in text
    assert 'Observation:' in text
    assert '![IMAGEGEN]' in text
