import copy
import os

import gradio as gr
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
from modelscope_agent.tools.openapi_plugin import OpenAPIPluginTool


# init user chatbot_agent
def init_user_chatbot_agent(uuid_str=''):
    builder_cfg, model_cfg, tool_cfg, available_tool_list, plugin_cfg, available_plugin_list = parse_configuration(
        uuid_str)
    # set top_p and stop_words for role play
    model_cfg[builder_cfg.model]['generate_cfg']['top_p'] = 0.5
    model_cfg[builder_cfg.model]['generate_cfg']['stop'] = 'Observation'

    # build model
    print(f'using model {builder_cfg.model}')
    print(f'model config {model_cfg[builder_cfg.model]}')
    try:
        llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)
    except Exception as e:
        raise gr.Error(str(e))

    # build prompt with zero shot react template
    instruction_template = parse_role_config(builder_cfg)
    prompt_generator = CustomPromptGenerator(
        system_template=DEFAULT_SYSTEM_TEMPLATE,
        user_template=DEFAULT_USER_TEMPLATE,
        exec_template=DEFAULT_EXEC_TEMPLATE,
        instruction_template=instruction_template,
        add_addition_round=True,
        addition_assistant_reply='好的。',
        knowledge_file_name=os.path.basename(builder_cfg.knowledge[0] if len(
            builder_cfg.knowledge) > 0 else ''),
        uuid_str=uuid_str)

    # get knowledge
    # 开源版本的向量库配置
    model_id = 'damo/nlp_gte_sentence-embedding_chinese-base'
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

    additional_tool_list = add_openapi_plugin_to_additional_tool(
        plugin_cfg, available_plugin_list)
    # build agent
    agent = AgentExecutor(
        llm,
        additional_tool_list=additional_tool_list,
        tool_cfg=tool_cfg,
        agent_type=AgentType.MRKL,
        prompt_generator=prompt_generator,
        knowledge_retrieval=knowledge_retrieval,
        tool_retrieval=False)
    agent.set_available_tools(available_tool_list + available_plugin_list)
    return agent


def add_openapi_plugin_to_additional_tool(plugin_cfgs, available_plugin_list):
    additional_tool_list = {}
    for name, cfg in plugin_cfgs.items():
        openapi_plugin_object = OpenAPIPluginTool(name=name, cfg=plugin_cfgs)
        additional_tool_list[name] = openapi_plugin_object
    return additional_tool_list


def user_chatbot_single_run(query, agent):
    agent.run(query)
