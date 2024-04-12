import logging
import os
import re
import time

import json
import ray
from modelscope_agent import create_component
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.agents import MultiRolePlay
from modelscope_agent.multi_agents_utils.executors.ray import RayTaskExecutor
from modelscope_agent.task_center import TaskCenter

REMOTE_MODE = True

# instruction prompt
ROLE_INSTRUCTION_PROMPT = """<|im_start|>system
你是{role}，请你根据对话情节设定、对话角色设定，继续当前的对话，推动剧情发展。

【对话情节设定】
{story}

【注意事项】
1. 根据user的历史对话回复时，不要重复说历史对话里的句子，要保持生成内容的有趣，丰富度，多样性
2. 长话短说，不要说太多话，不要超过50字

【你的角色设定】
{role_description}
<|im_end|>
"""

CHATROOM_INSTRUCTION_PROMPT = """<|im_start|>system
你是一个小说作家，请你根据对话场景、人物介绍及最近的对话记录，选择继续对话的下一个角色。注意，对话历史是以群聊的形式展现，因此角色可能会@某个人表示对这个人说话。

【对话场景】
{story}

【人物介绍】
{all_roles_info}

【对话记录】
chat_records

【最新对话】
recent_records

【当前对话主角】
{user_role}

【注意事项】
根据对话记录和最新对话
1. 当主角明确@某个角色，你需要让被@的角色接话。
2. 不要选【最新对话】里的角色发言
3. 要让每个角色有平等的对话机会，多选一些【对话记录里】没有出现的角色，要求情节充满戏剧性。
4. 尽量多的选择主角，让主角参与其中，当前对话的主角是{user_role}

【回复格式】
请用json格式回复，从上文提到的角色里选，只写名字即可，字段包括：
* next_speakers: <next speakers>
<|im_end|>
"""

STORY = """用户是男主角顾易，与六位长相、性格都大相径庭的美女相识，包括魅惑魔女郑梓妍、知性姐姐李云思、清纯女生肖鹿、刁蛮大小姐沈彗星、性感辣妈林乐清、冷艳总裁钟甄。
这六位美女都喜欢顾易，相互之间争风吃醋，展开一段轻喜甜蜜的恋爱之旅。
"""

# 用户列表
# roles = {
#     '顾易': '男主角，与六位美女相识，被美女包围，展开一段轻喜甜蜜的恋爱之旅',
#     '郑梓妍': '郑梓妍，23岁，射手座，A型血，鬼点子大王，极致魅惑，职业：杂志编辑',
#     '李云思': '李云思，27岁，摩羯座，O型血，趣味相投的知音，温婉大气，职业：策展人',
#     '肖鹿': '肖鹿， 20岁，天蝎座，B型血，脾气超大的实习生，小太阳，纯真无邪 ',
#     '沈彗星': '沈彗星，23岁， 白羊座，O型血，与顾易是青梅竹马，偶像剧女主角气质，刁蛮任性的傲娇千金',
#     '林乐清': '林乐清，28岁， 巨蟹座，B型血，合约女友，厨艺达人，性感火辣的斩男女神，离异性感辣妈， 职业：会计',
#     '钟甄': '钟甄，32岁， 狮子座，AB型血，负责任的女总裁，高贵冷艳的霸道女总，职业：会计事务所合伙人'
# }

STORY = """用户是男主角顾易，与多位长相、性格都大相径庭的美女相识，这几位美女都喜欢顾易，相互之间争风吃醋，展开一段轻喜甜蜜的恋爱之旅，除了顾易之外，其他几个角色之间一定要互相冲突打击对方。
"""

roles = {
    '顾易': '男主角，与多位美女相识，被美女包围，展开一段轻喜甜蜜的恋爱之旅',
    '郑梓妍': '郑梓妍，23岁，射手座，A型血，鬼点子大王，极致魅惑，职业：杂志编辑',
    '李云思': '李云思，27岁，摩羯座，O型血，趣味相投的知音，温婉大气，职业：策展人',
}

STORY = """用户是男主角雷军，小米创始人，最近发布了小米SU7，与多位新能源的创始人都认识，大家都在竞争卖车，雷军公布价格后，他们都很慌，害怕自己的车卖不出去。"""

roles = {
    '雷军': '小米创始人，最近发布了小米SU7，投资界之王，有很强的人缘，投资了很多公司。小米SU7的口号是【人车合一，我心澎湃】',
    '李想': '理想汽车，是由李想在2015年7月创立的新能源汽车公司，公司最初命名为“车和家，卖的最好的是理想L系列”',
    '李斌': '蔚来汽车创始人、董事长，也是新能源的早期创始人之一，之前做易车网',
    '何小鹏': '何小鹏同时担任小鹏汽车的董事长。小鹏汽车成立于2014年，小鹏汽车最热销的系列是小鹏P7，受影响最大，所以此次小米发布价格对其冲击很大'
}

llm_config = {
    #'model': 'qwen-max',
    'model': 'qwen-spark-plus',
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
for ctx, cur_role in enumerate(roles):
    all_roles_info += f'* 角色{ctx}: {cur_role}, {roles[cur_role]}\n'


def get_avatar_by_name(role_name):
    # get current file path
    current_file_path = os.path.abspath(__file__)

    # get current parent directory
    parent_directory_path = os.path.dirname(current_file_path)
    file_map = {
        '顾易': 'guyi.png',
        '郑梓妍': 'zhengziyan.png',
        '李云思': 'liyunsi.png',
        'others': 'default_girl.png'
    }
    if role_name not in file_map.keys():
        file_name = file_map['others']
    else:
        file_name = file_map[role_name]
    avatar_abs_path = os.path.join(parent_directory_path, 'resources',
                                   file_name)
    return avatar_abs_path


def generate_role_instruction(role):
    role_description = roles[role]
    instruction = ROLE_INSTRUCTION_PROMPT.format(
        role=role,
        #all_roles_info=all_roles_info,
        role_description=role_description,
        story=STORY)
    return instruction


def upsert_role(new_user, user_char, human_input_mode):
    role = create_component(
        MultiRolePlay,
        name=new_user,
        remote=REMOTE_MODE,
        role=new_user,
        description=user_char,
        llm=llm_config,
        function_list=function_list,
        instruction=generate_role_instruction(new_user),
        human_input_mode=human_input_mode)
    return role


def change_user_role(user_role, state):
    old_user_role = state['user_role']
    if user_role != old_user_role:
        task_center = state['task_center']
        task_center.reset_env()
        if REMOTE_MODE:
            ray.get(
                task_center.agent_registry.set_user_agent.remote(
                    old_user_role, 'CLOSE'))
            ray.get(
                task_center.agent_registry.set_user_agent.remote(
                    user_role, 'ON'))

        else:
            task_center.agent_registry.set_user_agent.remote(
                old_user_role, 'CLOSE')
            task_center.agent_registry.set_user_agent.remote(user_role, 'ON')
        state['task_center'] = task_center
        return state


def init_all_remote_actors(_roles, user_role, _state):
    RayTaskExecutor.init_ray()
    # if initialized, just change the user role
    if 'init' in _state and _state['init']:
        state = change_user_role(user_role, _state)
        return state

    task_center = create_component(
        TaskCenter, name='Task_Center', remote=REMOTE_MODE)

    # init all agents and task center
    role_agents = []
    for role in _roles:
        human_input_mode = 'CLOSE'
        if role == user_role:
            human_input_mode = 'ON'
        role_agent = upsert_role(role, _roles[role], human_input_mode)

        role_agents.append(role_agent)

    chat_room = create_component(
        MultiRolePlay,
        name='chat_room',
        remote=True,
        role='chat_room',
        llm=llm_config,
        function_list=function_list,
        instruction=CHATROOM_INSTRUCTION_PROMPT.format(
            all_roles_info=all_roles_info, story=STORY, user_role=user_role),
        is_watcher=True,
        use_history=False)

    logging.warning(msg=f'time:{time.time()} done create task center')

    ray.get(task_center.add_agents.remote(role_agents))
    ray.get(task_center.add_agents.remote([chat_room]))

    _state['agents'] = role_agents
    _state['task_center'] = task_center
    _state['user_role'] = user_role
    _state['role_names'] = list(_roles.keys())
    _state['init'] = True
    _state['next_agent_names'] = []
    return _state


def start_chat_with_topic(from_user, topic, _state):
    task_center = _state['task_center']
    ray.get(task_center.send_task_request.remote(topic, send_from=from_user))
    _state['task_center'] = task_center
    return _state


def chat_progress(user_response, _state):
    task_center = _state['task_center']
    role_names = _state['role_names']
    last_round_roles = _state['next_agent_names']

    while True:
        # get last round roles with user in order to process message from input
        if len(last_round_roles) > 0:
            for frame in task_center.step.remote(
                    allowed_roles=last_round_roles,
                    user_response=user_response,
                    **kwargs):
                yield ray.get(frame)

        # reset the last_round_roles to empty
        last_round_roles = []

        # chat_room decide the next speakers
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

            # only none user agent could send message here
            if len(user_agent_names) == 1:
                next_agent_names = list(
                    set(next_agent_names) - set(user_agent_names))

            # only if other agent than user run into this logic
            if len(next_agent_names) > 0:
                for frame in task_center.step.remote(
                        allowed_roles=next_agent_names,
                        user_response=user_response,
                        **kwargs):
                    yield ray.get(frame)

            # stop and let user send message
            if len(user_agent_names) == 1:
                next_agent_names_string = json.dumps(
                    {'next_agent_names': user_agent_names})
                yield next_agent_names_string
                break
        yield 'new_round'
