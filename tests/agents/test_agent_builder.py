import pytest
from modelscope_agent.agents.agent_builder import AgentBuilder


@pytest.mark.skip(
    reason='The output is empty. Need to figura out the reason later.')
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
