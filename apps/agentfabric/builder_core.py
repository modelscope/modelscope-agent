# flake8: noqa E501
import re
import traceback
from typing import Dict

import json
from config_utils import parse_configuration
from help_tools import LogoGeneratorTool, config_conversion
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MessagesGenerator

SYSTEM = 'You are a helpful assistant.'

LOGO_TOOL_NAME = 'logo_designer'

ANSWER = 'Answer'
CONFIG = 'Config'
ASSISTANT_PROMPT = """{}: <answer>\n{}: <config>\nRichConfig: <rich_config>""".format(
    ANSWER, CONFIG)

UPDATING_CONFIG_STEP = '🚀Updating Config...'
CONFIG_UPDATED_STEP = '✅Config Updated!'
UPDATING_LOGO_STEP = '🚀Updating Logo...'
LOGO_UPDATED_STEP = '✅Logo Updated!'


def init_builder_chatbot_agent(uuid_str):
    # build model
    builder_cfg, model_cfg, _, _, _, _ = parse_configuration(uuid_str)

    # additional tool
    additional_tool_list = {LOGO_TOOL_NAME: LogoGeneratorTool()}
    tool_cfg = {LOGO_TOOL_NAME: {'is_remote_tool': True}}

    # build llm
    print(f'using builder model {builder_cfg.model}')
    llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)
    llm.set_agent_type(AgentType.Messages)

    # build prompt
    # prompt generator
    prompt_generator = 'BuilderPromptGenerator'
    language = builder_cfg.get('language', 'en')
    if language == 'zh':
        prompt_generator = 'ZhBuilderPromptGenerator'

    # build agent
    agent = BuilderChatbotAgent(
        llm,
        tool_cfg,
        agent_type=AgentType.Messages,
        additional_tool_list=additional_tool_list,
        prompt_generator=prompt_generator)
    agent.set_available_tools([LOGO_TOOL_NAME])
    return agent


class BuilderChatbotAgent(AgentExecutor):

    def __init__(self, llm, tool_cfg, agent_type, additional_tool_list,
                 **kwargs):

        super().__init__(
            llm,
            tool_cfg,
            agent_type=agent_type,
            additional_tool_list=additional_tool_list,
            tool_retrieval=False,
            **kwargs)

        # used to reconstruct assistant message when builder config is updated
        self._last_assistant_structured_response = {}

    def stream_run(self,
                   task: str,
                   remote: bool = True,
                   print_info: bool = False,
                   uuid_str: str = '') -> Dict:

        # retrieve tools
        tool_list = self.retrieve_tools(task)
        self.prompt_generator.init_prompt(task, tool_list, [], self.llm.model)
        function_list = []

        llm_result, exec_result = '', ''

        idx = 0

        while True:
            idx += 1
            llm_artifacts = self.prompt_generator.generate(
                llm_result, exec_result)
            if print_info:
                print(f'|LLM inputs in round {idx}:\n{llm_artifacts}')

            llm_result = ''
            try:
                parser_obj = AnswerParser()
                for s in self.llm.stream_generate(llm_artifacts=llm_artifacts):
                    llm_result += s
                    answer, finish = parser_obj.parse_answer(llm_result)
                    if answer == '':
                        continue
                    result = {'llm_text': answer}
                    if finish:
                        result.update({'step': UPDATING_CONFIG_STEP})
                    yield result

                if print_info:
                    print(f'|LLM output in round {idx}:\n{llm_result}')
            except Exception as e:
                yield {'error': 'llm result is not valid'}

            try:
                re_pattern_config = re.compile(
                    pattern=r'Config: ([\s\S]+)\nRichConfig')
                res = re_pattern_config.search(llm_result)
                if res is None:
                    return
                config = res.group(1).strip()
                self._last_assistant_structured_response['config_str'] = config

                rich_config = llm_result[llm_result.rfind('RichConfig:')
                                         + len('RichConfig:'):].strip()
                try:
                    answer = json.loads(rich_config)
                except Exception:
                    print('parse RichConfig error')
                    return
                self._last_assistant_structured_response[
                    'rich_config_dict'] = answer
                builder_cfg = config_conversion(answer, uuid_str=uuid_str)
                yield {'exec_result': {'result': builder_cfg}}
                yield {'step': CONFIG_UPDATED_STEP}
            except ValueError as e:
                print(e)
                yield {'error content=[{}]'.format(llm_result)}
                return

            # record the llm_result result
            _ = self.prompt_generator.generate(
                {
                    'role': 'assistant',
                    'content': llm_result
                }, '')

            messages = self.prompt_generator.history
            if 'logo_prompt' in answer and len(messages) > 4 and (
                    answer['logo_prompt'] not in messages[-3]['content']):
                #  draw logo
                yield {'step': UPDATING_LOGO_STEP}
                params = {
                    'user_requirement': answer['logo_prompt'],
                    'uuid_str': uuid_str
                }

                tool = self.tool_list[LOGO_TOOL_NAME]
                try:
                    exec_result = tool(**params, remote=remote)
                    yield {'exec_result': exec_result}
                    yield {'step': LOGO_UPDATED_STEP}

                    return
                except Exception as e:
                    exec_result = f'Action call error: {LOGO_TOOL_NAME}: {params}. \n Error message: {e}'
                    yield {'error': exec_result}
                    self.prompt_generator.reset()
                    return
            else:
                return

    def update_config_to_history(self, config: Dict):
        """ update builder config to message when user modify configuration

        Args:
            config info read from builder config file
        """
        if len(
                self.prompt_generator.history
        ) > 0 and self.prompt_generator.history[-1]['role'] == 'assistant':
            answer = self._last_assistant_structured_response['answer_str']
            simple_config = self._last_assistant_structured_response[
                'config_str']

            rich_config_dict = {
                k: config[k]
                for k in ['name', 'description', 'prompt_recommend']
            }
            rich_config_dict[
                'logo_prompt'] = self._last_assistant_structured_response[
                    'rich_config_dict']['logo_prompt']
            rich_config_dict['instructions'] = config['instruction'].split('；')

            rich_config = json.dumps(rich_config_dict, ensure_ascii=False)
            new_content = ASSISTANT_PROMPT.replace('<answer>', answer).replace(
                '<config>', simple_config).replace('<rich_config>',
                                                   rich_config)
            self.prompt_generator.history[-1]['content'] = new_content


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


class AnswerParser(object):

    def __init__(self):
        self._history = ''

    def parse_answer(self, llm_result: str):
        finish = False
        answer_prompt = ANSWER + ': '

        if len(llm_result) >= len(answer_prompt):
            start_pos = llm_result.find(answer_prompt)
            end_pos = llm_result.find(f'\n{CONFIG}')
            if start_pos >= 0:
                if end_pos > start_pos:
                    result = llm_result[start_pos + len(answer_prompt):end_pos]
                    finish = True
                else:
                    result = llm_result[start_pos + len(answer_prompt):]
            else:
                result = llm_result
        else:
            result = ''

        new_result = result[len(self._history):]
        self._history = result
        return new_result, finish
