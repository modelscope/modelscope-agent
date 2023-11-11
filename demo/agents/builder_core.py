import copy
import os

from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MrklPromptGenerator
from modelscope_agent.retrieve import KnowledgeRetrieval

from modelscope.utils.config import Config

DEFAULT_BUILDER_CONFIG_FILE = "builder_config.json"
DEFAULT_MODEL_CONFIG_FILE = "model_config.json"
DEFAULT_TOOL_CONFIG_FILE = "tool_config.json"


def parse_configuration():
    """parse configuration

    Args:

    Returns:
        dict: parsed configuration

    """
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE', DEFAULT_MODEL_CONFIG_FILE)
    builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                 DEFAULT_BUILDER_CONFIG_FILE)
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)

    builder_cfg = Config.from_file(builder_cfg_file)
    model_cfg = Config.from_file(model_cfg_file)
    tool_cfg = Config.from_file(tool_cfg_file)

    tools_info = builder_cfg.tools
    available_tool_list = []
    for key, value in tools_info.items():
        if value['use']:
            available_tool_list.append(key)

    return builder_cfg, model_cfg, tool_cfg, available_tool_list


# put all the builder agent logic here


# init user chatbot_agent
def init_user_chatbot_agent():
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
    )

    # build model
    llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)

    # build prompt with zero shot react template
    tool_list_template = """Answer the following questions as best you can. You have access to the following tools:
    \n<tool_list>"""
    prompt_generator = MrklPromptGenerator(system_template='\n'.join(
        [builder_cfg.instruction, tool_list_template]))

    # get knowledge
    # 开源版本的向量库配置
    model_id = 'damo/nlp_corom_sentence-embedding_chinese-base'
    embeddings = ModelScopeEmbeddings(model_id=model_id)
    knowledge_retrieval = KnowledgeRetrieval.from_file(builder_cfg.knowledge,
                                                       embeddings, FAISS)

    # build agent
    agent = AgentExecutor(
        llm,
        tool_cfg,
        agent_type=AgentType.MRKL,
        prompt_generator=prompt_generator,
        knowledge_retrieval=knowledge_retrieval,
        tool_retrieval=False)
    agent.set_available_tools(available_tool_list)

    return agent


# TODO execute the user chatbot with user input in gradio
def execute_user_chatbot(*inputs):
    user_input = inputs[0]
    chatbot = inputs[1]
    # state = inputs[2]
    output_component = list(inputs[3:])

    for frame in chatbot_agent.stream_run(user_input, remote=True):
        # is_final = frame.get("frame_is_final")
        llm_result = frame.get("llm_text", "")
        exec_result = frame.get('exec_result', '')
        print(frame)
        llm_result = llm_result.split("<|user|>")[0].strip()
        if len(exec_result) != 0:
            # llm_result
            # update_component(exec_result)
            frame_text = ' '
        else:
            # action_exec_result
            frame_text = llm_result
        response = f'{response}\n{frame_text}'

        chatbot[-1] = (user_input, response)
        yield chatbot, *copy.deepcopy(output_component)


def user_chatbot_single_run(query, agent):
    agent.run(query)
