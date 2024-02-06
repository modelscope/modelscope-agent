import asyncio
import logging
import os
import time

import ray
from modelscope_agent.agents import RolePlay
from modelscope_agent.env_context_util import AgentEnvContextMixin
from modelscope_agent.environment import Environment

llm_config = {
    'model': 'qwen-max',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
    'model_server': 'dashscope'
}
function_list = []

# ray.init(local_mode=True)
if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)

env = Environment.create_remote(Environment)

role_play1 = AgentEnvContextMixin.create_remote(
    RolePlay,
    role='role_play1',
    env=env,
    llm=llm_config,
    function_list=function_list)

role_play2 = AgentEnvContextMixin.create_remote(
    RolePlay,
    role='role_play2',
    env=env,
    llm=llm_config,
    function_list=function_list)

ray.get(env.add_agents.remote([role_play1, role_play2]))

n_round = 1
task = 'who are u'
while n_round > 0:

    for frame in env.step.remote(task):
        print(ray.get(frame))

    env_list = env.get_message_list.remote('role_play1')
    queue = ray.get(env_list)
    x = len(queue)
    print(x)

    time.sleep(3)
    print('here3')
    n_round -= 1

# env_queue = env.get_message_queue_persist.remote('role_play1')
# queue = ray.get(env_queue)
# y = queue.get_nowait()
# print(y)

ray.shutdown()
