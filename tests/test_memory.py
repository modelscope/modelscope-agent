import sys
sys.path.append('../')
from modelscope_agent.prompt import PromptGenerator
def test_stm():
    res= PromptGenerator(user_template="""<|user|>:<user_input>""").init_prompt("你好吗",{},{})
    res = res.replace("\n","").strip()
    assert res == "这里是参考的历史信息,若没有信息则无需参考。这里是多轮对话历史。<|user|>:你好吗"

def test_ltm():
    #TODO run modelscope-agent/demo/demo_qwen_agent.ipynb to test it(print_info = True),second run will see the ltm in the prompt
    pass 
if __name__ == "__main__":
    test_stm()
