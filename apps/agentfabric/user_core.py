import os
import ssl

import gradio as gr
import nltk
from config_utils import parse_configuration
from custom_prompt import CustomPromptGenerator
from custom_prompt_zh import ZhCustomPromptGenerator
from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from modelscope_agent import prompt_generator_register
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.retrieve import KnowledgeRetrieval
from modelscope_agent.tools.openapi_plugin import OpenAPIPluginTool
from modelscope_agent.utils.logger import agent_logger as logger

prompts = {
    'CustomPromptGenerator': CustomPromptGenerator,
    'ZhCustomPromptGenerator': ZhCustomPromptGenerator,
}
prompt_generator_register(prompts)

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# init user chatbot_agent
def init_user_chatbot_agent(uuid_str=''):
    builder_cfg, model_cfg, tool_cfg, available_tool_list, plugin_cfg, available_plugin_list = parse_configuration(
        uuid_str)
    # set top_p and stop_words for role play
    if 'generate_cfg' not in model_cfg[builder_cfg.model]:
        model_cfg[builder_cfg.model]['generate_cfg'] = dict()
    model_cfg[builder_cfg.model]['generate_cfg']['top_p'] = 0.5
    model_cfg[builder_cfg.model]['generate_cfg']['stop'] = 'Observation'

    # build model
    logger.info(
        uuid=uuid_str,
        message=f'using model {builder_cfg.model}',
        content={'model_config': model_cfg[builder_cfg.model]})

    # # check configuration
    # if builder_cfg.model in ['qwen-max', 'qwen-72b-api', 'qwen-14b-api', 'qwen-plus']:
    #     if 'DASHSCOPE_API_KEY' not in os.environ:
    #         raise gr.Error('DASHSCOPE_API_KEY should be set via setting environment variable')

    try:
        llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)
    except Exception as e:
        raise gr.Error(str(e))

    # build prompt with zero shot react template
    prompt_generator = builder_cfg.get('prompt_generator', None)
    if builder_cfg.model.startswith('qwen') and not prompt_generator:
        prompt_generator = 'CustomPromptGenerator'
        language = builder_cfg.get('language', 'en')
        if language == 'zh':
            prompt_generator = 'ZhCustomPromptGenerator'

    prompt_cfg = {
        'prompt_generator':
        prompt_generator,
        'add_addition_round':
        True,
        'knowledge_file_name':
        os.path.basename(builder_cfg.knowledge[0]
                         if len(builder_cfg.knowledge) > 0 else ''),
        'uuid_str':
        uuid_str
    }

    # get knowledge
    # 开源版本的向量库配置
    model_id = 'damo/nlp_gte_sentence-embedding_chinese-base'
    embeddings = ModelScopeEmbeddings(model_id=model_id)
    available_knowledge_list = []
    for item in builder_cfg.knowledge:
        # if isfile and end with .txt, .md, .pdf, support only those file
        if os.path.isfile(item) and item.endswith(('.txt', '.md', '.pdf')):
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
        knowledge_retrieval=knowledge_retrieval,
        tool_retrieval=False,
        **prompt_cfg)
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
