from modelscope_agent.agents.function_calling import FunctionCalling


def test_function_calling_method():
    llm_config = {'model': 'qwen-turbo', 'model_server': 'dashscope'}

    # input tool name
    function_list = ['image_gen']

    bot = FunctionCalling(function_list=function_list, llm=llm_config)

    response = bot.run('画一只猫')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
    assert 'Action:' in text
    assert 'Action Input:' in text
    assert 'Observation::' in text
    assert 'Thought:' in text
    assert 'Final Answer:' in text
