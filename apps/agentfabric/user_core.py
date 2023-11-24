import copy
import os

from config_utils import parse_configuration
from custom_prompt import (DEFAULT_EXEC_TEMPLATE, DEFAULT_SYSTEM_TEMPLATE,
                           DEFAULT_USER_TEMPLATE, CustomPromptGenerator,
                           parse_role_config)
from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.retrieve import KnowledgeRetrieval


# init user chatbot_agent
def init_user_chatbot_agent(uuid_str=''):
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
        uuid_str)

    # build model
    print(f'using model {builder_cfg.model}')
    llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)

    # build prompt with zero shot react template
    instruction_template = parse_role_config(builder_cfg)
    prompt_generator = CustomPromptGenerator(
        system_template=DEFAULT_SYSTEM_TEMPLATE,
        user_template=DEFAULT_USER_TEMPLATE,
        exec_template=DEFAULT_EXEC_TEMPLATE,
        instruction_template=instruction_template,
        add_addition_round=True,
        addition_assistant_reply='好的。',
        knowledge_file_name=os.path.basename(builder_cfg.knowledge[0]),
        uuid_str=uuid_str)

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


def user_chatbot_single_run(query, agent):
    agent.run(query)
