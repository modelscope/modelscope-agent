import os.path
from typing import List

from config_utils import DEFAULT_UUID_HISTORY, parse_configuration
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.memory import FileStorageMemory
from modelscope_agent.tools.base import TOOL_REGISTRY
from modelscope_agent.tools.openapi_plugin import OpenAPIPluginTool
from modelscope_agent.utils.logger import agent_logger as logger


# init user chatbot_agent
def init_user_chatbot_agent(uuid_str='', session='default'):
    builder_cfg, model_cfg, tool_cfg, _, plugin_cfg, _ = parse_configuration(
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

    # update function_list
    function_list = parse_tool_cfg(tool_cfg)
    function_list = add_openapi_plugin_to_additional_tool(
        plugin_cfg, function_list)
    llm_config = {'model': builder_cfg.model, 'model_server': 'dashscope'}
    agent = RolePlay(function_list=function_list, llm=llm_config)

    # build memory
    current_history_path = os.path.join(DEFAULT_UUID_HISTORY, uuid_str,
                                        session + '_user.json')
    memory = FileStorageMemory(path=current_history_path)

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
