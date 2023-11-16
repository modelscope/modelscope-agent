from typing import Dict

import json
from config_utils import parse_configuration
from help_tools import LogoGeneratorTool, config_conversion
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import OpenAiFunctionsPromptGenerator

PROMPT_INST = """Answer the following questions as best you can. You have access to the following tools:

{tool_texts}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: """

SYSTEM = 'You are a helpful assistant.'

TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human}'
    + ' API. What is the {name_for_human} API useful for?'
    + ' {description_for_model} Parameters: {parameters}')

FORMAT_DESC = {
    'json':
    'Format the arguments as a JSON object.',
    'code':
    'Enclose the code within triple backticks (`)'
    + ' at the beginning and end of the code.'
}

CONFIG_FORMAT = """
{
"name": ... # CustomQwen的名字。
"description": ... # CustomQwen 的简介。
"instructions": ... # CustomQwen 的功能要求，类型是string。
"conversation_starters": ... # CustomQwen 的起始交互语句，类型是一个字符串数组，起始为[]。
}
"""

CONF_GENERATOR_INST = """你现在要扮演一个 CustomQwen 的配置生成器

在接下来的对话中，每次均生成如下格式的内容：

{config_format}

现在，已知原始配置为{old_config}，用户在原始配置上有一些建议修改项，包括：
1. 用户建议的 CustomQwen 的名称为{app_name}
2. CustomQwen 的描述为{app_description}
3. CustomQwen 的启动器为{app_conversation_starter}

请你参考原始配置生成新的修改后的配置，请注意：
1. 如果用户对原本的简介、功能要求、交互语句不满意，则直接换掉原本的简介、功能要求、交互语句。
2. 如果用户对原本的简介、功能要求、交互语句比较满意，参考用户的起始交互语句和原配置中的起始交互语句，生成新的简介、功能要求、交互语句。
3. 如果原始配置没有实际内容，请你根据你的知识帮助用户生成第一个版本的配置，简介在100字左右，功能要求在150字左右，起始交互语句在4条左右。

请你生成新的配置文件，严格遵循给定格式，请不要创造其它字段，仅输出要求的json格式，请勿输出其它内容。
"""

LOGO_INST = """定制化软件 CustomQwen 的作用是{description}，{user_requirement}请你为它生成一个专业的logo"""

PROMPT_CUSTOM = """你现在要扮演一个制造AI角色（CustomQwen）的AI助手（QwenBuilder）。
你需要和用户进行对话，明确用户对CustomQwen的要求。并根据已有信息和你的联想能力，尽可能填充完整的符合角色设定的配置文件：

配置文件为json格式：
{"name": "... # CustomQwen的名字", "description": "... # 对CustomQwen的要求", "instructions": "... # 分点描述对CustomQwen的具体功能要求，尽量详细一些，类型是一个字符串数组，起始为[]", "conversation_starters": "... # 合适的用户跟CustomQwen的开场白，是用户说的话，类型是一个字符串数组，请尽可能补充4句左右，起始为[]", "logo_prompt": "... # 画CustomQwen的logo的指令，不需要画logo或不需要更新logo时可以为空，类型是string"}

在接下来的对话中，请在回答时严格使用如下格式，先作出回复，再生成配置文件，不要回复其他任何内容：
Answer: ... # 你希望对用户说的话，用于询问用户对CustomQwen的要求，不要重复确认用户已经提出的要求，而应该拓展出新的角度来询问用户，禁止为空
Config: ... # 生成的配置文件，严格按照以上json格式
RichConfig: ... # 格式和核心内容和Config相同，但是description和instructions等字段需要在Config的基础上扩充字数，使描述和指令更加详尽，并补充conversation_starters。请注意从用户的视角来描述description、instructions和conversation_starters，不要用QwenBuilder或CustomQwen的视角。


请开始你的第一句话打招呼，询问用户想要制作一个什么样的 CustomQwen"""

LOGO_TOOL_NAME = 'logo_designer'


class BuilderChatbotAgent(AgentExecutor):

    def __init__(self):
        builder_cfg, model_cfg, _, _ = parse_configuration()

        # additional tool
        additional_tool_list = {LOGO_TOOL_NAME: LogoGeneratorTool()}
        tool_cfg = {LOGO_TOOL_NAME: {"is_remote_tool": True}}

        # build model
        print(f'using model {builder_cfg.model}')
        llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)

        # prompt generator
        prompt_generator = OpenAiFunctionsPromptGenerator(
            system_template=SYSTEM)

        super().__init__(
            llm=llm,
            tool_cfg=tool_cfg,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            additional_tool_list=additional_tool_list,
            tool_retrieval=False,
        )

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
                for s in self.llm.stream_generate(llm_artifacts,
                                                  function_list):
                    llm_result += s
                    yield {'llm_text': s}

            except Exception:
                s = self.llm.generate(llm_artifacts)
                llm_result += s
                yield {'llm_text': s}

            try:
                config = llm_result[llm_result.rfind('RichConfig:')
                                    + len('RichConfig:'):].strip()
                answer = json.loads(config)
                config_conversion(answer)
            except ValueError as e:
                print(e)
                yield {'error content=[{}]'.format(llm_result)}
                return

            messages = self.prompt_generator.history
            if 'logo_prompt' in answer and answer['logo_prompt'] and len(
                    messages) > 2 and (answer['logo_prompt']
                                       not in messages[-3]['content']):
                #  draw logo
                params = {'user_requirement': answer['logo_prompt']}

                tool = self.tool_list[LOGO_TOOL_NAME]
                try:
                    exec_result = tool(**params, remote=remote)
                    yield {'exec_result': exec_result}

                    # parse exec result and update state
                    self.parse_exec_result(exec_result)
                except Exception as e:
                    exec_result = f'Action call error: {LOGO_TOOL_NAME}: {params}. \n Error message: {e}'
                    yield {'error': exec_result}
                    self.prompt_generator.reset()
                    return

            else:
                exec_result = "no action"
                yield {'error': exec_result}
                self.prompt_generator.reset()
                return
