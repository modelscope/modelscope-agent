from modelscope.utils.config import Config

cfg = Config.from_file('config/cfg_tool_template.json')


def test_modelscope_speech_generation():
    from modelscope_agent.tools.modelscope_tools import TexttoSpeechTool
    kwargs = """{'input': '北京今天天气怎样?', 'gender': 'man'}"""
    txt2speech = TexttoSpeechTool(cfg)
    res = txt2speech.call(kwargs)
    print(res)


def test_modelscope_video_generation():
    from modelscope_agent.tools.modelscope_tools import TextToVideoTool
    kwargs = """{'text': '一个正在打篮球的人'}"""
    txt2video = TextToVideoTool(cfg)
    res = txt2video.call(kwargs)
    print(res)


def test_modelscope_text_address():
    from modelscope_agent.tools.modelscope_tools import TextAddressTool
    kwargs = """{'input': '北京朝阳望京东金辉大厦'}"""
    txt_addr = TextAddressTool(cfg)
    res = txt_addr.call(kwargs)
    print(res)


def test_modelscope_text_ner_remote():
    from modelscope_agent.tools.modelscope_tools import TextNerTool
    kwargs = """{'input': '多数新生儿甲亢在出生时即有症状，表现为突眼、甲状腺肿大、烦躁、多动、心动过速、呼吸急促，严重可出现心力衰竭，血T3、T4升高，TSH下降。'}"""
    txt_ner = TextNerTool()
    res = txt_ner.call(kwargs)
    print(res)


def test_modelscope_text_ner_local():
    from modelscope_agent.tools.modelscope_tools import TextNerTool
    kwargs = """{'input': '多数新生儿甲亢在出生时即有症状，表现为突眼、甲状腺肿大、烦躁、多动、心动过速、呼吸急促，严重可出现心力衰竭，血T3、T4升高，TSH下降。'}"""
    cfg = {
        'text-ner': {
            'is_remote_tool': False,
        }
    }
    txt_ner = TextNerTool(cfg)
    res = txt_ner.call(kwargs)
    print(res)


def test_modelscope_text_ie():
    from modelscope_agent.tools.modelscope_tools import TextInfoExtractTool
    kwargs = """{'input': '很满意，音质很好，发货速度快，值得购买', 'schema': {'属性词': {'情感词': null}}}"""
    txt_ie = TextInfoExtractTool(cfg)
    res = txt_ie.call(kwargs)
    print(res)


def test_modelscope_en2zh():
    from modelscope_agent.tools.modelscope_tools import TranslationEn2ZhTool
    kwargs = """{'input': 'Autonomous agents have long been a prominent research focus in both academic and industry communities.'}"""  # noqa E501
    zh_txt = TranslationEn2ZhTool(cfg)
    res = zh_txt.call(kwargs)
    print(res)


def test_modelscope_zh2en():
    from modelscope_agent.tools.modelscope_tools import TranslationZh2EnTool
    kwargs = """{'input': '北京今天天气怎样?'}"""
    en_txt = TranslationZh2EnTool(cfg)
    res = en_txt.call(kwargs)
    print(res)


def test_modelscope_image_chat():
    from modelscope_agent.tools.modelscope_tools import ImageChatTool
    kwargs = """{'image': 'http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/portrait_input.png', 'text': 'Describe the facial expression of the man.'}"""  # noqa E501
    image_chat = ImageChatTool(cfg)
    res = image_chat.call(kwargs)
    print(res)
