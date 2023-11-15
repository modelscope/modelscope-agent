import copy
import os

from custom_prompt import (DEFAULT_EXEC_TEMPLATE, DEFAULT_SYSTEM_TEMPLATE,
                           DEFAULT_USER_TEMPLATE)
from help_tools import ConfGeneratorTool, LogoGeneratorTool
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
    print(f'using model {builder_cfg.model}')
    llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)

    # build prompt with zero shot react template
    prompt_generator = MrklPromptGenerator(
        system_template=DEFAULT_SYSTEM_TEMPLATE,
        user_template=DEFAULT_USER_TEMPLATE,
        exec_template=DEFAULT_EXEC_TEMPLATE,
        instruction_template=builder_cfg.instruction,
    )

    # get knowledge
    # 开源版本的向量库配置
    model_id = 'damo/nlp_corom_sentence-embedding_chinese-base'
    embeddings = ModelScopeEmbeddings(model_id=model_id)
    available_knowledge_list = []
    for item in builder_cfg.knowledge:
        if os.path.isfile(item):
            available_knowledge_list.append(item)
    if len(available_knowledge_list) > 0:
        knowledge_retrieval = KnowledgeRetrieval.from_file(
            available_knowledge_list, embeddings, FAISS)
    else:
        knowledge_retrieval = None

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
def init_builder_chatbot_agent():
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
    )

    # build tool
    additional_tool_list = {
        'LogoGenerator': LogoGeneratorTool({'is_remote_tool': True}),
        'ConfGenerator': ConfGeneratorTool({'is_remote_tool': True})
    }

    # build model
    llm = LLMFactory.build_llm(builder_cfg.builder_model, model_cfg)

    # build prompt with zero shot react template
    prompt_generator = MrklPromptGenerator()

    agent = AgentExecutor(
        llm,
        agent_type=AgentType.MRKL,
        prompt_generator=prompt_generator,
        additional_tool_list=additional_tool_list,
        tool_retrieval=False)
    agent.set_available_tools(additional_tool_list.keys())

    return agent


def user_chatbot_single_run(query, agent):
    agent.run(query)
