from config_utils import parse_configuration
from modelscope_agent.llm import LLMFactory


def init_builder_chatbot_agent():
    builder_cfg, model_cfg, tool_cfg, available_tool_list = parse_configuration(
    )

    # build model
    print(f'using model {builder_cfg.model}')
    # llm = LLMFactory.build_llm(builder_cfg.model, model_cfg)
