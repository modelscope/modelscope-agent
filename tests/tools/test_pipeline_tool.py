from modelscope_agent.tools.pipeline_tool import ModelscopePipelineTool
from modelscope.utils.config import Config
import os

cfg = Config.from_file('config/cfg_tool_template.json')
# 请用自己的SDK令牌替换{YOUR_MODELSCOPE_SDK_TOKEN}（包括大括号）
os.environ['MODELSCOPE_API_KEY'] = f"{YOUR_MODELSCOPE_SDK_TOKEN}"

def test_modelscope_speech_generation():
    from modelscope_agent.tools.text_to_speech_tool import TexttoSpeechTool
    kwargs = """{'input': '北京今天天气怎样?', 'gender': 'man'}"""
    txt2speech = TexttoSpeechTool(cfg)
    res = txt2speech.call(kwargs)
    print(res)


test_modelscope_speech_generation()

