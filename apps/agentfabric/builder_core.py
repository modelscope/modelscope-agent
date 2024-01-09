# flake8: noqa E501
import os
import re

import json
from config_utils import DEFAULT_UUID_HISTORY, parse_configuration
from help_tools import config_conversion, logo_generate_remote_call
from modelscope_agent.agents import AgentBuilder
from modelscope_agent.memory import FileStorageMemory
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger

LOGO_TOOL_NAME = 'logo_designer'

UPDATING_CONFIG_STEP = '🚀Updating Config...'
CONFIG_UPDATED_STEP = '✅Config Updated!'
UPDATING_LOGO_STEP = '🚀Updating Logo...'
LOGO_UPDATED_STEP = '✅Logo Updated!'


def init_builder_chatbot_agent(uuid_str: str, session='default'):
    # read config
    # Todo: how to load the config?
    builder_cfg, model_cfg, _, _, _, _ = parse_configuration(uuid_str)

    # init agent
    logger.info(
        uuid=uuid_str, message=f'using builder model {builder_cfg.model}')
    llm_config = {'model': builder_cfg.model, 'model_server': 'dashscope'}
    # function_list = ['image_gen']  # use image_gen to draw logo?

    agent = AgentBuilder(llm=llm_config)

    current_history_path = os.path.join(DEFAULT_UUID_HISTORY, uuid_str,
                                        session + '_builder.json')
    memory = FileStorageMemory(path=current_history_path)

    return agent, memory


def gen_response_and_process(agent,
                             query: str,
                             memory: FileStorageMemory,
                             uuid_str: str,
                             print_info: bool = False):
    """
    process the response of one QA for the agent
    this need be in an agent, but the response format is not Union[str, Iterator[str]]
    """
    history = memory.get_history()
    llm_result = ''
    llm_result_prefix = ''
    try:
        response = agent.run(query, history=history)
        for s in response:
            llm_result += s
            answer, finish, llm_result_prefix = agent.parse_answer(
                llm_result_prefix, llm_result)
            if answer == '':
                continue
            result = {
                'llm_text': answer
            }  # Incremental content in streaming output
            if finish:
                result.update({'step': UPDATING_CONFIG_STEP})
            yield result

        # update memory
        if len(history) == 0:
            memory.update_history(
                Message(role='system', content=agent.system_prompt))
        memory.update_history([
            Message(role='user', content=query),
            Message(role='assistant', content=llm_result),
        ])
        if print_info:
            logger.info(
                uuid=uuid_str,
                message=f'LLM output in round 0',
                content={'llm_result': llm_result})
    except Exception as e:
        yield {'error': 'llm result is not valid'}

    try:
        re_pattern_config = re.compile(
            pattern=r'Config: ([\s\S]+)\nRichConfig')
        res = re_pattern_config.search(llm_result)
        if res is None:
            return
        config = res.group(1).strip()
        agent.last_assistant_structured_response['config_str'] = config

        rich_config = llm_result[llm_result.rfind('RichConfig:')
                                 + len('RichConfig:'):].strip()
        try:
            answer = json.loads(rich_config)
        except Exception:
            logger.error(uuid=uuid_str, error='parse RichConfig error')
            return
        agent.last_assistant_structured_response['rich_config_dict'] = answer
        builder_cfg = config_conversion(answer, uuid_str=uuid_str)
        yield {'exec_result': {'result': builder_cfg}}
        yield {'step': CONFIG_UPDATED_STEP}
    except ValueError as e:
        logger.error(uuid=uuid_str, error=str(e))
        yield {'error content=[{}]'.format(llm_result)}
        return
    agent.messages.append({
        'role': 'user',
        'content': query,
    })
    agent.messages.append({
        'role': 'assistant',
        'content': llm_result,
    })

    if 'logo_prompt' in answer:
        if not history or answer['logo_prompt'] not in history[-1]['content']:
            # draw logo
            yield {'step': UPDATING_LOGO_STEP}
            params = {
                'user_requirement': answer['logo_prompt'],
                'uuid_str': uuid_str
            }

            try:
                exec_result = logo_generate_remote_call(**params)
                yield {'exec_result': exec_result}
                yield {'step': LOGO_UPDATED_STEP}

                return
            except Exception as e:
                exec_result = f'Action call error: {LOGO_TOOL_NAME}: {params}. \n Error message: {e}'
                yield {'error': exec_result}
                # self.prompt_generator.reset()
                return
        else:
            return


def beauty_output(response: str, step_result: str):
    flag_list = [
        CONFIG_UPDATED_STEP, UPDATING_CONFIG_STEP, LOGO_UPDATED_STEP,
        UPDATING_LOGO_STEP
    ]

    if step_result in flag_list:
        end_str = ''
        for item in flag_list:
            if response.endswith(item):
                end_str = item
        if end_str == '':
            response = f'{response}\n{step_result}'
        elif end_str in [CONFIG_UPDATED_STEP, LOGO_UPDATED_STEP]:
            response = f'{response}\n{step_result}'
        else:
            response = response[:-len('\n' + end_str)]
            response = f'{response}\n{step_result}'

    return response
