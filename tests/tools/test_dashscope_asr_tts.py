import pytest
from modelscope_agent.agents.role_play import RolePlay  # NOQA
from modelscope_agent.tools import ParaformerAsrTool, SambertTtsTool


@pytest.mark.skip()
def test_paraformer_asr():
    params = """{'audio_path': '34aca18b-17a1-4558-9064-22fdfcef7a94.wav'}"""
    asr_tool = ParaformerAsrTool()
    res = asr_tool.call(params)
    assert res == 'today is a beautiful day. '


@pytest.mark.skip()
def test_sambert_tts():
    params = """{'text': '今天天气怎么样？'}"""
    tts_tool = SambertTtsTool()
    res = tts_tool.call(params)
    assert res.endswith('.wav')


@pytest.mark.skip()
def test_paraformer_asr_agent():
    role_template = '你扮演一个语音专家，用尽可能丰富的描述调用工具处理语音。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    function_list = ['paraformer_asr']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run(
        '[上传文件 "34aca18b-17a1-4558-9064-22fdfcef7a94.wav"], 将上面的音频识别出来')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)


def test_sambert_tts_agent():
    role_template = '你扮演一个语音专家，能够调用工具合成语音。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    function_list = ['sambert_tts']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('合成语音，语音内容为：“今天天气怎么样？会下雨吗？”')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
