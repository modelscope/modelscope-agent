from modelscope_agent.agent import AgentExecutor
from modelscope_agent.tools import ParaformerAsrTool
from modelscope_agent.tools import SambertTtsTool
from tests.utils import MockLLM, MockOutParser, MockPromptGenerator


def test_paraformer_asr():
    kwargs = {'audio_path': '16k-xwlb3_local_user.wav'}
    asr_tool = ParaformerAsrTool()
    res = asr_tool.__call__(**kwargs)
    print(res['result'])


def test_sambert_tts():
    kwargs = {'text': '今天天气怎么样？'}
    tts_tool = SambertTtsTool()
    res = tts_tool.__call__(**kwargs)
    print(res['result'])


def test_paraformer_asr_agent():
    responses = [
        "<|startofthink|>{\"api_name\": \"paraformer_asr_utils\", \"parameters\": "
        "{\"audio_path\": \"16k-xwlb3_local_user.wav\"}}<|endofthink|>", 'summarize'
    ]
    llm = MockLLM(responses)

    tools = {'paraformer_asr_utils': ParaformerAsrTool()}
    prompt_generator = MockPromptGenerator()
    action_parser = MockOutParser('paraformer_asr_utils',
                                  {'audio_path': '16k-xwlb3_local_user.wav'})

    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        action_parser=action_parser,
        tool_retrieval=False,
    )
    res = agent.run('将上面的音频识别出来')
    print(res)


def test_sambert_tts_agent():
    responses = [
        "<|startofthink|>{\"api_name\": \"sambert_tts_utils\", \"parameters\": "
        "{\"text\": \"今天天气怎么样？会下雨吗？\"}}<|endofthink|>", 'summarize'
    ]
    llm = MockLLM(responses)

    tools = {'sambert_tts_utils': SambertTtsTool()}
    prompt_generator = MockPromptGenerator()
    action_parser = MockOutParser('sambert_tts_utils',
                                  {'text': '今天天气怎么样？会下雨吗？'})

    agent = AgentExecutor(
        llm,
        additional_tool_list=tools,
        prompt_generator=prompt_generator,
        action_parser=action_parser,
        tool_retrieval=False,
    )
    res = agent.run('合成一段语音')
    print(res)
