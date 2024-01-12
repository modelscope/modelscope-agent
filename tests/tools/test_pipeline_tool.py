from modelscope.utils.config import Config
from modelscope_agent.tools.utils.output_wrapper import AudioWrapper
from modelscope_agent.tools.utils.output_wrapper import VideoWrapper
import os
import difflib

cfg = Config.from_file('config/cfg_tool_template.json')


def test_modelscope_speech_generation_remote():
    from modelscope_agent.tools.modelscope_tools import TexttoSpeechTool
    kwargs = """{'input': '北京今天天气怎样?', 'voice': 'zhizhe_emo'}"""
    txt2speech = TexttoSpeechTool()
    res = txt2speech.call(kwargs)
    assert isinstance(res, AudioWrapper), "结果不是AudioWrapper类型"
    print(res)

def test_modelscope_speech_generation_local():
    from modelscope_agent.tools.modelscope_tools import TexttoSpeechTool
    kwargs = """{'input': '北京今天天气怎样?', 'voice': 'zhizhe_emo'}"""
    cfg = {
        'speech-generation': {
            'is_remote_tool': False,
        }
    }
    txt2speech = TexttoSpeechTool(cfg)
    res = txt2speech.call(kwargs)
    assert (res.startswith("b'RIFF"))


def test_modelscope_video_generation_remote():
    from modelscope_agent.tools.modelscope_tools import TextToVideoTool
    kwargs = """{'text': 'A person who is playing basketball'}"""
    txt2video = TextToVideoTool()
    res = txt2video.call(kwargs)
    assert isinstance(res, VideoWrapper), "结果不是VideoWrapper类型"
    print(res)

def test_modelscope_video_generation_local():
    from modelscope_agent.tools.modelscope_tools import TextToVideoTool
    kwargs = """{'text': 'A person who is playing basketball'}"""
    cfg = {
        'video-generation': {
            'is_remote_tool': False,
        }
    }
    txt2video = TextToVideoTool(cfg)
    res = txt2video.call(kwargs)
    assert (res.startswith("b"))


def test_modelscope_text_address_remote():
    from modelscope_agent.tools.modelscope_tools import TextAddressTool
    kwargs = """{'input': '北京朝阳望京东金辉大厦'}"""
    txt_addr = TextAddressTool()
    res = txt_addr.call(kwargs)
    assert (res.startswith("[{'end'"))
    print(res)

def test_modelscope_text_address_local():
    from modelscope_agent.tools.modelscope_tools import TextAddressTool
    kwargs = """{'input': '北京朝阳望京东金辉大厦'}"""
    cfg = {
        'text-address': {
            'is_remote_tool': False,
        }
    }
    txt_addr = TextAddressTool(cfg)
    res = txt_addr.call(kwargs)
    assert (res.startswith("[{'type'"))
    print(res)


def test_modelscope_text_ner_remote():
    from modelscope_agent.tools.modelscope_tools import TextNerTool
    kwargs = """{'input': '多数新生儿甲亢在出生时即有症状，表现为突眼、甲状腺肿大、烦躁、多动、心动过速、呼吸急促，严重可出现心力衰竭，血T3、T4升高，TSH下降。'}"""
    txt_ner = TextNerTool()
    res = txt_ner.call(kwargs)
    assert (res.startswith("[{'end':"))
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
    assert (res.startswith("[{'type':"))
    print(res)


def test_modelscope_text_ie_remote():
    from modelscope_agent.tools.modelscope_tools import TextInfoExtractTool
    kwargs = """{'input': '很满意，音质很好，发货速度快，值得购买', 'schema': {'属性词': {'情感词': null}}}"""
    txt_ie = TextInfoExtractTool()
    res = txt_ie.call(kwargs)
    assert (res.startswith("[[{'offset'"))
    print(res)

def test_modelscope_text_ie_local():
    from modelscope_agent.tools.modelscope_tools import TextInfoExtractTool
    kwargs = """{'input': '很满意，音质很好，发货速度快，值得购买', 'schema': {'属性词': {'情感词': null}}}"""
    cfg = {
        'text-ie': {
            'is_remote_tool': False,
        }
    }
    txt_ie = TextInfoExtractTool(cfg)
    res = txt_ie.call(kwargs)
    assert (res.startswith("[[{'type'"))
    print(res)


def test_modelscope_en2zh_remote():
    from modelscope_agent.tools.modelscope_tools import TranslationEn2ZhTool
    kwargs = """{'input': 'Autonomous agents have long been a prominent research focus in both academic and industry communities.'}"""  # noqa E501
    zh_txt = TranslationEn2ZhTool()
    res = zh_txt.call(kwargs)

    example = "长期以来 ， 自主代理一直是学术界和工业界的重要研究重点。"
    similarity = difflib.SequenceMatcher(None, res, example).ratio()
    similarity_threshold = 0.8
    assert similarity >= similarity_threshold, f"相似度为 {similarity:.2%}，未达到期望的相似度要求。"

    print(res)

def test_modelscope_en2zh_local():
    from modelscope_agent.tools.modelscope_tools import TranslationEn2ZhTool
    kwargs = """{'input': 'Autonomous agents have long been a prominent research focus in both academic and industry communities.'}"""  # noqa E501
    cfg = {
        'text-translation-en2zh': {
            'is_remote_tool': False,
        }
    }
    zh_txt = TranslationEn2ZhTool(cfg)
    res = zh_txt.call(kwargs)

    example = "长期以来 ， 自主代理一直是学术界和工业界的重要研究重点。"
    similarity = difflib.SequenceMatcher(None, res, example).ratio()
    similarity_threshold = 0.8
    assert similarity >= similarity_threshold, f"相似度为 {similarity:.2%}，未达到期望的相似度要求。"

    print(res)


def test_modelscope_zh2en_remote():
    from modelscope_agent.tools.modelscope_tools import TranslationZh2EnTool
    kwargs = """{'input': '北京今天天气怎样?'}"""
    en_txt = TranslationZh2EnTool()
    res = en_txt.call(kwargs)

    example = "What's the weather like in Beijing today?"
    similarity = difflib.SequenceMatcher(None, res, example).ratio()
    similarity_threshold = 0.8
    assert similarity >= similarity_threshold, f"相似度为 {similarity:.2%}，未达到期望的相似度要求。"

    print(res)

def test_modelscope_zh2en_local():
    from modelscope_agent.tools.modelscope_tools import TranslationZh2EnTool
    kwargs = """{'input': '北京今天天气怎样?'}"""
    cfg = {
        'text-translation-zh2en': {
            'is_remote_tool': False,
        }
    }
    en_txt = TranslationZh2EnTool(cfg)
    res = en_txt.call(kwargs)

    example = "What's the weather like in Beijing today?"
    similarity = difflib.SequenceMatcher(None, res, example).ratio()
    similarity_threshold = 0.8
    assert similarity >= similarity_threshold, f"相似度为 {similarity:.2%}，未达到期望的相似度要求。"

    print(res)


def test_modelscope_image_chat_remote():
    from modelscope_agent.tools.modelscope_tools import ImageChatTool
    kwargs = """{'image': 'http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/portrait_input.png', 'text': 'Describe the facial expression of the man.'}"""  # noqa E501
    image_chat = ImageChatTool()
    res = image_chat.call(kwargs)

    example = "The man has a very angry facial expression, with his mouth wide open and his eyes wide"
    similarity = difflib.SequenceMatcher(None, res, example).ratio()
    similarity_threshold = 0.8
    assert similarity >= similarity_threshold, f"相似度为 {similarity:.2%}，未达到期望的相似度要求。"

    print(res)

def test_modelscope_image_chat_local():
    from modelscope_agent.tools.modelscope_tools import ImageChatTool
    kwargs = """{'image': 'http://mm-chatgpt.oss-cn-zhangjiakou.aliyuncs.com/mplug_owl_demo/released_checkpoint/portrait_input.png', 'text': 'Describe the facial expression of the man.'}"""  # noqa E501
    cfg = {
        'image-chat': {
            'is_remote_tool': False,
        }
    }
    image_chat = ImageChatTool(cfg)
    res = image_chat.call(kwargs)

    example = "The man has a very angry facial expression, with his mouth wide open and his eyes wide"
    similarity = difflib.SequenceMatcher(None, res, example).ratio()
    similarity_threshold = 0.8
    assert similarity >= similarity_threshold, f"相似度为 {similarity:.2%}，未达到期望的相似度要求。"

    print(res)


# speech
test_modelscope_speech_generation_remote()
test_modelscope_speech_generation_local()

# video generation
test_modelscope_video_generation_remote()
test_modelscope_video_generation_local()

# ner
test_modelscope_text_ner_local()
test_modelscope_text_ner_remote()

# image chat
test_modelscope_image_chat_local()
test_modelscope_image_chat_remote()

# text address
test_modelscope_text_address_remote()
test_modelscope_text_address_local()

# text_ie
test_modelscope_text_ie_remote()
test_modelscope_text_ie_local()

# en2zh
test_modelscope_en2zh_remote()
test_modelscope_en2zh_local()

# zh2en
test_modelscope_zh2en_remote()
test_modelscope_zh2en_local()