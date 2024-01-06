import pytest
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import PromptGenerator
from modelscope_agent.prompt.prompt_factory import get_prompt_generator
from tests.utils import MockLLM, MockPromptGenerator, MockTool

from modelscope.utils.config import Config

model_cfg_file = 'config/cfg_model_template.json'
model_cfg = Config.from_file(model_cfg_file)


def test_qwen_zh():
    cfg = {'language': 'zh'}
    model_id = 'qwen_7b_dashscope'
    llm = LLMFactory.build_llm(model_id, model_cfg)
    agent = AgentExecutor(llm, **cfg)

    from modelscope_agent.prompt import MrklPromptGenerator
    assert isinstance(agent.prompt_generator, MrklPromptGenerator)


def test_qwen_default():
    model_id = 'qwen_plus'
    llm = LLMFactory.build_llm(model_id, model_cfg)
    agent = AgentExecutor(llm)

    from modelscope_agent.prompt import MessagesGenerator
    assert isinstance(agent.prompt_generator, MessagesGenerator)


def test_chatglm():
    model_id = 'chatglm3-6b-dashscope'
    llm = LLMFactory.build_llm(model_id, model_cfg)
    agent = AgentExecutor(llm)

    from modelscope_agent.prompt import ChatGLMPromptGenerator
    assert isinstance(agent.prompt_generator, ChatGLMPromptGenerator)


def test_gpt():
    model_id = 'openai'
    llm = LLMFactory.build_llm(model_id, model_cfg)
    agent = AgentExecutor(llm)

    from modelscope_agent.prompt import MessagesGenerator
    assert isinstance(agent.prompt_generator, MessagesGenerator)
