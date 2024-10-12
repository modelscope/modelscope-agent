import copy
import os.path
from typing import List

import json
from config_utils import get_user_preview_history_dir, parse_configuration
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.memory import MemoryWithRag
from modelscope_agent.tools.base import TOOL_REGISTRY
from modelscope_agent.tools.openapi_plugin import OpenAPIPluginTool
from modelscope_agent.utils.logger import agent_logger as logger


# init user chatbot_agent
def init_user_chatbot_agent(uuid_str='',
                            session='default',
                            use_tool_api=False,
                            user_token=None):
    builder_cfg, model_cfg, tool_cfg, _, openapi_plugin_cfg, openapi_plugin_list = parse_configuration(
        uuid_str, use_tool_api)
    # set top_p and stop_words for role play
    if 'generate_cfg' not in model_cfg[builder_cfg.model]:
        model_cfg[builder_cfg.model]['generate_cfg'] = dict()
    model_cfg[builder_cfg.model]['generate_cfg']['top_p'] = 0.5
    model_cfg[builder_cfg.model]['generate_cfg']['stop'] = 'Observation'

    # update function_list
    function_list = parse_tool_cfg(tool_cfg)

    if not use_tool_api:
        function_list = add_openapi_plugin_to_additional_tool(
            openapi_plugin_cfg, function_list)

    # build model
    logger.query_info(
        uuid=uuid_str,
        message=
        f'using model {builder_cfg.model} with tool {tool_cfg} and function list {function_list}',
        details={'model_config': model_cfg[builder_cfg.model]})

    llm_config = copy.deepcopy(model_cfg[builder_cfg.model])
    llm_config['model_server'] = llm_config.pop('type')
    instruction = {
        'name': builder_cfg.name,
        'description': builder_cfg.description,
        'instruction': builder_cfg.instruction
    }
    agent = RolePlay(
        function_list=function_list,
        llm=llm_config,
        instruction=instruction,
        uuid_str=uuid_str,
        use_tool_api=use_tool_api,
        user_token=user_token,
        openapi_list=openapi_plugin_list)

    # build memory
    preview_history_dir = get_user_preview_history_dir(uuid_str, session)
    storage_path = preview_history_dir
    memory_history_path = os.path.join(preview_history_dir,
                                       session + '_user.json')
    memory_agent_name = uuid_str + '_' + session + '_memory'
    memory = MemoryWithRag(
        urls=builder_cfg.knowledge,
        llm=agent.llm,
        storage_path=storage_path,
        name=memory_agent_name,
        memory_path=memory_history_path,
        use_knowledge_cache=False,
    )

    return agent, memory


def parse_tool_cfg(tool_cfg):
    tool_cfg_in_dict = tool_cfg.to_dict()
    function_list = [
        key for key, value in tool_cfg_in_dict.items()
        if value.get('use') and value.get('is_active')
    ]
    return function_list


def add_openapi_plugin_to_additional_tool(plugin_cfgs, function_list):
    if plugin_cfgs is None or plugin_cfgs == {}:
        return function_list
    for name, _ in plugin_cfgs.items():
        openapi_plugin_object = OpenAPIPluginTool(name=name, cfg=plugin_cfgs)
        TOOL_REGISTRY[name] = openapi_plugin_object
        function_list.append(name)
    return function_list
