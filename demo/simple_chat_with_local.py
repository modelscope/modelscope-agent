import logging
import os
import time

from modelscope_agent import create_component
from modelscope_agent.agents import RolePlay
from modelscope_agent.task_center import TaskCenter

REMOTE_MODE = False

llm_config = {
    'model': 'qwen-max',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
    'model_server': 'dashscope'
}
function_list = []

task_center = create_component(
    TaskCenter, name='task_center', remote=REMOTE_MODE)

logging.warning(msg=f'time:{time.time()} done create task center')

role_play1 = create_component(
    RolePlay,
    name='role_play1',
    remote=REMOTE_MODE,
    llm=llm_config,
    function_list=function_list)

role_play2 = create_component(
    RolePlay,
    name='role_play2',
    remote=REMOTE_MODE,
    llm=llm_config,
    function_list=function_list)

task_center.add_agents([role_play1, role_play2])

n_round = 2
task = 'who are u'
task_center.send_task_request(task)
while n_round > 0:

    for frame in task_center.step():
        print(frame)

    time.sleep(3)

    n_round -= 1
