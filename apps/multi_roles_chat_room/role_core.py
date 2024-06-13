import logging
import os
import re
import time

import json
import ray
from modelscope_agent import create_component
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.agents.multi_role_play import MultiRolePlay
from modelscope_agent.task_center import TaskCenter
from story_holder import get_story_by_id

REMOTE_MODE = True

# instruction prompt
ROLE_INSTRUCTION_PROMPT = """<|im_start|>system
你是{role}，角色介绍：{role_description}

【对话场景】
{story}

【注意事项】
1. 长话短说，不要说太多话，不要超过50字
<|im_end|>"""

CHATROOM_INSTRUCTION_PROMPT = """<|im_start|>system
你现在是一个小说作家，请你根据对话场景、人物介绍及最近的对话记录，选择继续对话的下一个角色。注意，对话历史是以群聊的形式展现，因此角色可能会@某个人表示对这个人说话。

【对话场景】
{story}

【人物介绍】
{all_roles_info}

【chat history】
chat_records

【最新对话】
recent_records

【注意事项】
根据chat history和最新对话
1. 当主角{user_role}说话中提到某个角色，你需要只让提到的角色接话。
2. 不要选【最新对话】里的角色发言
3. 要让每个角色有平等的对话机会，多选一些chat history没有出现的角色，要求情节充满戏剧性。
4. 只写角色名字即可，每次最多选两个角色，尽量多的选择主角，当前对话的主角是{user_role}

【回复格式】
请用json格式回复，从上文提到的角色里选，只写名字即可，每次最多选两个角色，不要选太多，字段包括：
* next_speakers: <next speaker>
<|im_end|>
"""
llm_config = {
    # 'model': 'qwen-max',
    'model': 'qwen-spark-plus-0403',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
    'model_server': 'dashscope'
}
kwargs = {
    'temperature': 0.92,
    'top_p': 0.95,
    'seed': 1683806810,
}
function_list = []


def get_all_roles_info(roles):
    all_roles_info = ''
    for ctx, cur_role in enumerate(roles):
        all_roles_info += f'* 角色{ctx}: {cur_role}, {roles[cur_role]}\n'
    return all_roles_info


def generate_role_instruction(role, story_info):
    role_description = story_info['roles'][role]
    instruction = ROLE_INSTRUCTION_PROMPT.format(
        role=role,
        # all_roles_info=all_roles_info,
        role_description=role_description,
        story=story_info['story'])
    return instruction


def upsert_role(new_user, user_char, human_input_mode, story_info, llm_config,
                _uid):
    role = create_component(
        MultiRolePlay,
        name=new_user,
        remote=REMOTE_MODE,
        role=new_user,
        description=user_char,
        llm=llm_config,
        function_list=function_list,
        instruction=generate_role_instruction(new_user, story_info),
        human_input_mode=human_input_mode,
        prefix_name=_uid)
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


def init_all_remote_actors(_roles, user_role, _state, _story_state,
                           select_model, _uid):
    story_info = get_story_by_id(_story_state)
    llm_config['model'] = select_model

    # if initialized, just change the user role
    if 'init' in _state and _state['init']:
        state = change_user_role(user_role, _state)
        return state

    task_center = create_component(
        TaskCenter, name='Task_Center', remote=REMOTE_MODE, prefix_name=_uid)

    # init all agents and task center
    role_agents = []
    for role in _roles:
        human_input_mode = 'CLOSE'
        if role == user_role:
            human_input_mode = 'ON'
        role_agent = upsert_role(role, _roles[role], human_input_mode,
                                 story_info, llm_config, _uid)

        role_agents.append(role_agent)

    chat_room = create_component(
        MultiRolePlay,
        name='chat_room',
        remote=True,
        llm=llm_config,
        function_list=function_list,
        instruction=CHATROOM_INSTRUCTION_PROMPT.format(
            all_roles_info=get_all_roles_info(story_info['roles']),
            story=story_info['story'],
            user_role=user_role),
        is_watcher=True,
        use_history=False,
        prefix_name=_uid)

    logging.warning(
        msg=f'time:{time.time()} done create task center with uid: {_uid}')

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
