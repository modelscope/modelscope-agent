import logging
import os
import time

import ray
from modelscope_agent import create_component
from modelscope_agent.agents import RolePlay
from modelscope_agent.multi_agents_task import TaskCenter

llm_config = {
    'model': 'qwen-max',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
    'model_server': 'dashscope'
}
function_list = []

task_center = TaskCenter(remote=True)
logging.warning(msg=f'time:{time.time()} done create task center')

role_play1 = create_component(
    RolePlay,
    name='role_play1',
    remote=True,
    role='role_play1',
    llm=llm_config,
    function_list=function_list)

role_play2 = create_component(
    RolePlay,
    name='role_play2',
    remote=True,
    role='role_play2',
    llm=llm_config,
    function_list=function_list)

task_center.add_agents([role_play1, role_play2])

n_round = 2
task = 'who are u'
task_center.start_task(task)
while n_round > 0:

    for frame in TaskCenter.step.remote(task_center):
        print(ray.get(frame))

    time.sleep(3)

    n_round -= 1

ray.shutdown()
