from modelscope.utils.config import Config

cfg = Config.from_file('config/cfg_tool_template.json')


def test_modelscope_speech_generation():
    from modelscope_agent.tools import TexttoSpeechTool
    input = '北京今天天气怎样?'
    kwargs = {'input': input, 'gender': 'man'}
    txt2speech = TexttoSpeechTool(cfg)
    res = txt2speech._remote_call(**kwargs)

    print(res)


test_modelscope_speech_generation()


def test_modelscope_text_address():
    from modelscope_agent.tools import TextAddressTool
    input = '北京朝阳望京东金辉大厦'
    kwargs = {'input': input}
    txt_addr = TextAddressTool(cfg)
    res = txt_addr._remote_call(**kwargs)

    print(res)


def test_modelscope_text_ner():
    from modelscope_agent.tools import TextNerTool
    input = '北京今天天气怎样?'
    kwargs = {'input': input}
    txt_ner = TextNerTool(cfg)
    res = txt_ner._remote_call(**kwargs)

    print(res)


def test_modelscope_video_generation():
    from modelscope_agent.tools import TextToVideoTool
    input = '一个正在打篮球的人'
    kwargs = {'text': input}
    video_gen = TextToVideoTool(cfg)
    res = video_gen._remote_call(**kwargs)

    print(res)


def test_modelscope_zh2en():
    from modelscope_agent.tools import TranslationZh2EnTool
    input = '北京今天天气怎样?'
    kwargs = {'input': input}
    zh_to_en = TranslationZh2EnTool(cfg)
    res = zh_to_en._remote_call(**kwargs)

    print(res)


def test_modelscope_en2zh():
    from modelscope_agent.tools import TranslationEn2ZhTool
    input = 'Autonomous agents have long been a prominent research focus in both academic and industry communities.'
    kwargs = {'input': input}
    en_to_zh = TranslationEn2ZhTool(cfg)
    res = en_to_zh._remote_call(**kwargs)

    print(res)
