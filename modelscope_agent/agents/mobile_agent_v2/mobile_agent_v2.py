import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import json
from modelscope_agent import Agent
from modelscope_agent.environment import ADBEnvironment
from modelscope_agent.llm import get_chat_model
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.utils.logger import agent_logger as logger

from .prompt import (get_action_prompt, get_memory_prompt, get_process_prompt,
                     get_reflect_prompt, get_system_prompt)


class MobileAgentV2(Agent):

    def __init__(self,
                 env: ADBEnvironment,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm_planner: Optional[Union[Dict, BaseChatModel]] = None,
                 llm_decision: Optional[Union[Dict, BaseChatModel]] = None,
                 llm_reflect: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 **kwargs):

        self.env = env

        if isinstance(llm_planner, Dict):
            self.llm_config_planner = llm_planner
            self.llm_planner = get_chat_model(**self.llm_config_planner)
        else:
            self.llm_planner = llm_planner

        if isinstance(llm_decision, Dict):
            self.llm_config_decision = llm_decision
            self.llm_decision = get_chat_model(**self.llm_config_decision)
        else:
            self.llm_decision = llm_decision

        if isinstance(llm_reflect, Dict):
            self.llm_config_reflect = llm_reflect
            self.llm_reflect = get_chat_model(**self.llm_config_reflect)
        else:
            self.llm_reflect = llm_reflect
        self.stream = True

        self.function_list = []
        self.function_map = {}
        if function_list:
            for function in function_list:
                self._register_tool(function)

        self.storage_path = storage_path
        self.mem = None

    def _run(self, user_query, **kwargs):
        step = 0

        thought_history = []
        summary_history = []
        action_history = []
        add_info = "If you want to tap an icon of an app, use the action \"Open app\""
        summary = ''
        action = ''
        completed_requirements = ''
        memory = ''

        error_flag = False

        logger.info('Start running mobile agent')
        while True:
            step += 1
            this_results = {}
            logger.info(f'Oberserve the environment: Step {step}')
            perception_infos, width, height, keyboard, screenshot_file = self.env.observe(
            )

            # decision
            prompt_action = get_action_prompt(user_query, perception_infos,
                                              width, height, keyboard,
                                              summary_history, action_history,
                                              summary, action, add_info,
                                              error_flag,
                                              completed_requirements, memory)
            system_prompt_decision = get_system_prompt('decision')
            messages = [{'role': 'system', 'content': system_prompt_decision}]
            messages.append({'role': 'user', 'content': prompt_action})

            self._parse_image_url([screenshot_file], messages)

            logger.info(f'Call decision agent: Step {step}')
            output_decision = self.llm_decision.chat(messages=messages)
            # this_results['decision'] = output_action

            thought = output_decision.split(
                '### Thought ###')[-1].split('### Action ###')[0].replace(
                    '\n', ' ').replace(':', '').replace('  ', ' ').strip()
            summary = output_decision.split('### Operation ###')[-1].replace(
                '\n', ' ').replace('  ', ' ').strip()
            action = output_decision.split('### Action ###')[-1].split(
                '### Operation ###')[0].replace('\n',
                                                ' ').replace('  ',
                                                             ' ').strip()

            messages.append({'role': 'assistant', 'content': output_decision})

            prompt_memory = get_memory_prompt()

            messages.append({'role': 'user', 'content': prompt_memory})

            logger.info(f'Call decision agent with action: Step {step}')
            output_memory = self.llm_decision.chat(messages=messages)

            messages.append({'role': 'assistant', 'content': output_memory})

            output_memory = output_memory.split('### Important content ###')[
                -1].split('\n\n')[0].strip() + '\n'
            if 'None' not in output_memory and output_memory not in memory:
                memory += output_memory
                this_results['memory'] = output_memory

            print_status(messages)

            if self.env.act(action):
                break

            last_perception_infos = copy.deepcopy(perception_infos)
            last_keyboard = keyboard
            last_screenshot_file = screenshot_file

            logger.info(f'Observe the environment before reflect: Step {step}')
            perception_infos, width, height, keyboard, screenshot_file = self.env.observe(
            )

            # reflect
            prompt_reflect = get_reflect_prompt(
                user_query, last_perception_infos, perception_infos, width,
                height, last_keyboard, keyboard, summary, action, add_info)
            system_prompt_reflect = get_system_prompt('reflect')
            messages = [{'role': 'system', 'content': system_prompt_reflect}]

            messages.append({'role': 'user', 'content': prompt_reflect})

            self._parse_image_url([last_screenshot_file, screenshot_file],
                                  messages)

            logger.info(f'Call reflect agent: Step {step}')
            output_reflect = self.llm_reflect.chat(messages=messages)
            this_results['reflect'] = output_reflect
            reflect = output_reflect.split('### Answer ###')[-1].replace(
                '\n', ' ').strip()
            messages.append({'role': 'assistant', 'content': output_reflect})
            print_status(messages)

            if 'A' in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)

                prompt_memory = get_process_prompt(user_query, thought_history,
                                                   summary_history,
                                                   action_history,
                                                   completed_requirements,
                                                   add_info)
                system_prompy_plan = get_system_prompt('plan')
                messages = [{'role': 'system', 'content': system_prompy_plan}]
                messages.append({'role': 'user', 'content': prompt_memory})

                logger.info(f'Call planner agent: Step {step}')
                output_memory = self.llm_planner.chat(messages=messages)

                messages.append({
                    'role': 'assistant',
                    'content': output_memory
                })
                print_status(messages)

                completed_requirements = output_memory.split(
                    '### Completed contents ###')[-1].replace('\n',
                                                              ' ').strip()
                this_results['process'] = output_memory

                error_flag = False

            elif 'B' in reflect:
                error_flag = True
                self.env.act('Back')

            elif 'C' in reflect:
                error_flag = True


def print_status(chat_history):
    print('*' * 100)
    for chat in chat_history:
        print('role:', chat['role'])
        content = chat['content']
        if isinstance(content, str):
            print(content)
        else:
            print(content[0]['text'] + '<image>' * (len(content[1]) - 1)
                  + '\n')
    print('*' * 100)
