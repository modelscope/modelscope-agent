# flake8: noqa E501
import re
from typing import Dict

import json
from config_utils import parse_configuration
from help_tools import LogoGeneratorTool, config_conversion
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MessagesGenerator

SYSTEM = 'You are a helpful assistant.'

PROMPT_CUSTOM = """你现在要扮演一个制造AI角色（AI-Agent）的AI助手（QwenBuilder）。
你需要和用户进行对话，明确用户对AI-Agent的要求。并根据已有信息和你的联想能力，尽可能填充完整的符合角色设定的配置文件：

配置文件为json格式：
{"name": "... # AI-Agent的名字", "description": "... # 对AI-Agent的要求", "instructions": "... # 分点描述对AI-Agent的具体功能要求，尽量详细一些，类型是一个字符串数组，起始为[]", "conversation_starters": "... # 合适的用户跟AI-Agent的开场白，是用户说的话，类型是一个字符串数组，请尽可能补充4句左右，起始为[]", "logo_prompt": "... # 画AI-Agent的logo的指令，不需要画logo或不需要更新logo时可以为空，类型是string"}

在接下来的对话中，请在回答时严格使用如下格式，先作出回复，再生成配置文件，不要回复其他任何内容：
Answer: ... # 你希望对用户说的话，用于询问用户对AI-Agent的要求，不要重复确认用户已经提出的要求，而应该拓展出新的角度来询问用户，禁止为空
Config: ... # 生成的配置文件，严格按照以上json格式
RichConfig: ... # 格式和核心内容和Config相同，但是description和instructions等字段需要在Config的基础上扩充字数，使描述和指令更加详尽，并补充conversation_starters。请注意从用户的视角来描述description、instructions和conversation_starters，不要用QwenBuilder或AI-Agent的视角。


明白了请说“好的。”， 不要说其他的。"""

LOGO_TOOL_NAME = 'logo_designer'


def init_builder_chatbot_agent():
    # build model
    builder_cfg, model_cfg, _, _ = parse_configuration()

    # additional tool
    additional_tool_list = {LOGO_TOOL_NAME: LogoGeneratorTool()}
    tool_cfg = {LOGO_TOOL_NAME: {'is_remote_tool': True}}

    # build llm
    print(f'using builder model {builder_cfg.model}')
    llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)
    llm.set_agent_type(AgentType.Messages)

    # build prompt
    starter_messages = [{
        'role': 'system',
        'content': SYSTEM
    }, {
        'role': 'user',
        'content': PROMPT_CUSTOM
    }, {
        'role': 'assistant',
        'content': '好的。'
    }]

    # prompt generator
    prompt_generator = MessagesGenerator(
        system_template=SYSTEM, custom_starter_messages=starter_messages)

    # build agent
    agent = BuilderChatbotAgent(
        llm,
        tool_cfg,
        agent_type=AgentType.Messages,
        prompt_generator=prompt_generator,
        additional_tool_list=additional_tool_list)
    agent.set_available_tools([LOGO_TOOL_NAME])
    return agent


class BuilderChatbotAgent(AgentExecutor):

    def stream_run(self,
                   task: str,
                   remote: bool = True,
                   print_info: bool = False) -> Dict:

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
                # no stream yet
                llm_result = self.llm.generate(llm_artifacts)
                if print_info:
                    print(f'|LLM output in round {idx}:\n{llm_result}')

                re_pattern_answer = re.compile(
                    pattern=r'Answer:([\s\S]+)\nConfig:')
                res = re_pattern_answer.search(llm_result['content'])
                llm_text = res.group(1).strip()
                yield {'llm_text': llm_text}
            except Exception:
                yield {'error': 'llm result is not valid'}

            try:
                if self.agent_type == AgentType.Messages:
                    content = llm_result['content']
                else:
                    content = llm_result
                config = content[content.rfind('RichConfig:')
                                 + len('RichConfig:'):].strip()
                answer = json.loads(config)
                builder_cfg = config_conversion(answer)
                yield {'exec_result': {'result': builder_cfg}}
            except ValueError as e:
                print(e)
                yield {'error content=[{}]'.format(content)}
                return

            # record the llm_result result
            _ = self.prompt_generator.generate(llm_result, '')

            messages = self.prompt_generator.history
            if 'logo_prompt' in answer and len(messages) > 4 and (
                    answer['logo_prompt'] not in messages[-3]['content']):
                #  draw logo
                params = {'user_requirement': answer['logo_prompt']}

                tool = self.tool_list[LOGO_TOOL_NAME]
                try:
                    exec_result = tool(**params, remote=remote)
                    yield {'exec_result': exec_result}

                    return
                except Exception as e:
                    exec_result = f'Action call error: {LOGO_TOOL_NAME}: {params}. \n Error message: {e}'
                    yield {'error': exec_result}
                    self.prompt_generator.reset()
                    return
            else:
                return
