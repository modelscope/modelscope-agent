from modelscope_agent.agents.role_play import RolePlay  # NOQA
from modelscope_agent.tools import ParaformerAsrTool, SambertTtsTool


def test_paraformer_asr():
    params = """{'audio_path': '16k-xwlb3_local_user.wav'}"""
    asr_tool = ParaformerAsrTool()
    res = asr_tool.call(params)
    print(res['result'])
test_paraformer_asr()

def test_sambert_tts():
    params = """{'text': '今天天气怎么样？'}"""
    tts_tool = SambertTtsTool()
    res = tts_tool.call(params)
    print(res['result'])


def test_paraformer_asr_agent():
    role_template = '你扮演一个语音专家，用尽可能丰富的描述调用工具处理语音。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    function_list = ['paraformer_asr']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('[上传文件 "16k-xwlb3_local_user.wav"], 将上面的音频识别出来')

    text = ''
    for chunk in response:
        text += chunk
    print(text)


def test_sambert_tts_agent():
    role_template = '你扮演一个语音专家，用尽可能丰富的描述调用工具处理语音。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    function_list = ['paraformer_asr']

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('合成语音，语音内容为：“今天天气怎么样？会下雨吗？”')

    text = ''
    for chunk in response:
        text += chunk
    print(text)
