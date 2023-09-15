from modelscope_agent.llm import LLM

RESPONSE = [
    """非常好的大纲！接下来，我们来确定一些关键情节和转折。我建议一开始，小男孩和他的狗在家附近的森林中发现一个神秘的地图。他们决定追寻地图的线索，这将带领他们穿越森林，遇到各种困难和挑战。最后，他们找到了一份宝藏，但他们意识到真正的宝藏其实是他们之间的友谊。你觉得如何？""",
    """太好了！接下来，我建议这个故事的语言风格应该是生动而富有冒险精神的，你认同吗？""",
    # """明白了，我会以这种风格来生成故事。稍等一下，我马上生成故事。""",
    """在一天阳光明媚的早晨，小男孩Tommy和他的狗Max在后院发现了一个被风吹来的神秘地图。地图上标记着一条通向森林深处的路线，一颗大大的"X"标记在地图的另一头。
    "Max，看看这个！我们找到宝藏地图了！" Tommy兴奋地对他的狗说。他们决定冒险寻找宝藏。他们沿着地图上的路线，穿过森林，爬过山丘，甚至跨过一条吓人的独木桥。
    他们遇到了许多难关，但他们并没有放弃。他们一起解决问题，一起笑，一起哭。这次冒险使他们的友谊更加深厚。
    最后，他们在一座古老的洞穴里发现了宝藏，但他们意识到，真正的宝藏其实是他们经历的冒险，以及他们之间无可替代的友谊。
    <|startofthink|>```JSON\n{\n   "api_name": "print_story_tool",\n    "parameters": {\n      "text": "在一天阳光明媚的早晨，小男孩Tommy和他的狗Max在后院发现了一个被风吹来的神秘地图。地图上标记着一条通向森林深处的路线，一颗大大的X标记在地图的另一边"\n   }\n}\n```<|endofthink|>""",
    """test""",
    """很高兴你喜欢！接下来，我们可以开始创建故事的插图。你更喜欢哪种风格的插图？是手绘风格，还是数字艺术风格？
    <|startofthink|>```JSON\n{\n   "api_name": "show_image_example",\n    "parameters": {\n      "visible": "true"\n   }\n}\n```<|endofthink|>""",
    """test""",
    """好的，正在为你生成插图，请稍后。<|startofthink|>```JSON\n{\n   "api_name": "image_generation",\n    "parameters": {\n      "text": "A little boy and his dog find a mysterious map in their backyard", "idx": "0" \n   }\n}\n```<|endofthink|>""",
    """已经为你生成第一张图片，请稍后。<|startofthink|>```JSON\n{\n   "api_name": "image_generation",\n    "parameters": {\n      "text": "The little boy and his dog ventured through the forest, climbing hills and crossing wooden Bridges", "idx": "1" \n   }\n}\n```<|endofthink|>""",
    """test""",
]


class MockLLM(LLM):
    def __init__(self):
        cfg = {}
        super().__init__(cfg)
        self.response = RESPONSE
        self.idx = -1
    
    def generate(self, text):
        self.idx += 1
        return self.response[self.idx] if self.idx < len(self.response) else "default"