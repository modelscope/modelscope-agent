import logging
import os
import re
import time

import json
import ray
from modelscope_agent import create_component
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.agents import RolePlay
from modelscope_agent.multi_agents_utils.executors.ray import RayTaskExecutor
from modelscope_agent.task_center import TaskCenter

REMOTE_MODE = True

if REMOTE_MODE:
    RayTaskExecutor.init_ray()

ROLE_INSTRUCTION_PROMPT = """你是{role}，请你根据对话情节设定、对话角色设定，继续当前的对话，推动剧情发展。

# 对话情节设定
{story}

# 所有对话角色设定
{all_roles_info}

# 你的角色设定：
{role_description}

# 注意事项
1. 这是聊天室，不要发发送私信给任何人
2. 仅代表你个人说话,不要扮演其他人

"""

CHATROOM_INSTRUCTION_PROMPT = """你是一个小说作家，请你根据对话场景、人物介绍及最近的对话记录，选择继续对话的下一个角色。

# 对话场景
{story}

# 人物介绍
{all_roles_info}

# 注意事项
1. 所有角色不区分主角和配角，你需要让每个角色有平等的对话机会，要求情节充满戏剧性。
2. 当上一个角色已经连发多条消息，你需要让另一个角色接话。
3. 当主角明确@某个角色，你需要让被@的角色接话。
4. 允许下一个角色可以是多个人，不要超过3个，可以是1到3人，随机一些

# 回复格式
请用json格式回复，字段包括
* plot: <first summarize recent chat history in 20 words>
* thought: <think who is most likely to speak next>
* next_speakers: <next speakers>


# 最近对话历史
最近的对话历史会通过用户信息展示。


**不要返回任何json格式以外信息，包括```json ```这个格式，不要试图续写**
"""

STORY = """用户是男主角顾易，与六位长相、性格都大相径庭的美女相识，包括魅惑魔女郑梓妍、知性姐姐李云思、清纯女生肖鹿、刁蛮大小姐沈彗星、性感辣妈林乐清、冷艳总裁钟甄。
这六位美女都喜欢顾易，相互之间争风吃醋，展开一段轻喜甜蜜的恋爱之旅。
"""

ROLES_MAP = {
    '顾易': '男主角，与六位美女相识，被美女包围，展开一段轻喜甜蜜的恋爱之旅',
    '郑梓妍': '郑梓妍，23岁，射手座，A型血，鬼点子大王，极致魅惑，职业：杂志编辑',
    '李云思': '李云思，27岁，摩羯座，O型血，趣味相投的知音，温婉大气，职业：策展人',
    '肖鹿': '肖鹿， 20岁，天蝎座，B型血，脾气超大的实习生，小太阳，纯真无邪 ',
    '沈彗星': '沈彗星，23岁， 白羊座，O型血，与顾易是青梅竹马，偶像剧女主角气质，刁蛮任性的傲娇千金',
    '林乐清': '林乐清，28岁， 巨蟹座，B型血，合约女友，厨艺达人，性感火辣的斩男女神，离异性感辣妈， 职业：会计',
    '钟甄': '钟甄，32岁， 狮子座，AB型血，负责任的女总裁，高贵冷艳的霸道女总，职业：会计事务所合伙人'
}

llm_config = {
    'model': 'qwen-max',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
    'model_server': 'dashscope'
}
kwargs = {
    'temperature': 0.92,
    'top_p': 0.95,
    'seed': 1683806810,
}
function_list = []

all_roles_info = ''
for cur_role in ROLES_MAP:
    all_roles_info += f'* {cur_role}\n {ROLES_MAP[cur_role]}\n\n'


def generate_role_instruction(role):
    role_description = ROLES_MAP[role]
    instruction = ROLE_INSTRUCTION_PROMPT.format(
        role=role,
        all_roles_info=all_roles_info,
        role_description=role_description,
        story=STORY)
    return instruction


def init_all_agents():
    agents = []
    for role in ROLES_MAP:
        human_input_mode = 'CLOSE'
        if role == '顾易':
            human_input_mode = 'TERMINAL'

        agent = create_component(
            RolePlay,
            name=role,
            remote=REMOTE_MODE,
            description=ROLES_MAP[role],
            llm=llm_config,
            function_list=function_list,
            instruction=generate_role_instruction(role),
            human_input_mode=human_input_mode)
        agents.append(agent)
    return agents


task_center = create_component(
    TaskCenter, name='task_center', remote=REMOTE_MODE)

logging.warning(msg=f'time:{time.time()} done create task center')

roles = init_all_agents()

role_names = ROLES_MAP.keys()

chat_room = create_component(
    RolePlay,
    name='chat_room',
    remote=REMOTE_MODE,
    llm=llm_config,
    function_list=function_list,
    instruction=CHATROOM_INSTRUCTION_PROMPT.format(
        all_roles_info=all_roles_info, story=STORY),
    is_watcher=True,
    use_history=False)

ray.get(task_center.add_agents.remote(roles))
ray.get(task_center.add_agents.remote([chat_room]))

# start the chat by send chat message to env
chat = '@顾易 要不要来我家吃饭？'
ray.get(task_center.send_task_request.remote(chat, send_from='林乐清'))

# limit the round to n_round
n_round = 50

while n_round > 0:

    # decide the next speakers
    chat_room_result = ''
    for frame in task_center.step.remote(
            allowed_roles=['chat_room'], **kwargs):
        remote_result = ray.get(frame)
        print(remote_result)
        raw_result = AgentEnvMixin.extract_frame(remote_result)
        chat_room_result += raw_result['content']

    try:
        if chat_room_result.startswith('```json'):
            re_pattern_config = re.compile(pattern=r'```json([\s\S]+)```')
            res = re_pattern_config.search(chat_room_result)
            chat_room_result = res.group(1).strip()
        response_json = json.loads(chat_room_result)
        if isinstance(response_json['next_speakers'], str):
            next_agent_names = re.findall('|'.join(role_names),
                                          response_json['next_speakers'])
        else:
            next_agent_names = [
                item for item in response_json['next_speakers']
                if item in role_names
            ]
    except Exception:
        next_agent_names = []

    if len(next_agent_names) > 0:
        user_agent_names = ray.get(
            task_center.is_user_agent_present.remote(next_agent_names))

        if len(user_agent_names) == 1:
            # only one user agent in this case
            user_response = input(
                f'You are {user_agent_names}. Press enter to skip and use auto-reply, '
                f'or input any information to talk with other roles: ')

        for frame in task_center.step(
                allowed_roles=next_agent_names,
                user_response=user_response,
                **kwargs):
            print(ray.get(frame))

ray.shutdown()
