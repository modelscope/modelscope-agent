import os

import pytest
from modelscope_agent.agents.agent_builder import AgentBuilder

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_agent_builder():
    llm_config = {'model': 'qwen-turbo', 'model_server': 'dashscope'}

    # input tool name
    function_list = ['image_gen']

    bot = AgentBuilder(function_list=function_list, llm=llm_config)

    response = bot.run('创建一个多啦A梦')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
    assert 'Answer:' in text
    assert 'Config:' in text
    assert 'RichConfig:' in text
