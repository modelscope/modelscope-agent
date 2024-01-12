from modelscope_agent.tools import WordArtTexture

from modelscope_agent.agents.role_play import RolePlay  # NOQA


def test_word_art():
    params = """{
        'input.text.text_content': '魔搭社区',
        'input.prompt': '一片绿色的森林里开着小花',
        'input.texture_style': 'scene',
        'input.text.output_image_ratio': '9:16'
    }"""
    wa = WordArtTexture()
    res = wa.call(params)
    print(res)
    assert (res.startswith('http'))
