from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial

import gradio as gr
from dotenv import load_dotenv
from gradio.components import Chatbot as ChatBot
from help_tool import ImageGenerationTool, PrintStoryTool, ShowExampleTool
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator

from modelscope.utils.config import Config

PROMPT_START = "你好！我是你的StoryAgent，很高兴为你提供服务。首先，我想了解你对想要创作的故事有什么大概的想法或者大纲？"


SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Story Agent，不断和用户沟通创作故事想法，确定完后生成故事给用户，然后再询问用户绘图风格，最后生成绘图给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""

SYSTEM_PROMPT = ""

#<|assistant|>: 你好！我是你的StoryAgent，很高兴为你提供服务。首先，我想了解你对想要创作的故事有什么大概的想法或者大纲？
INSTRUCTION_TEMPLATE =  """\n\n<|system|>: 你现在扮演一个Story Agent，不断和用户沟通创作故事想法，确定完后生成故事给用户，然后再询问用户绘图风格，最后生成绘图给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
    【多轮对话历史】

    <|user|>: 给我生成一个关于友谊和冒险的故事，主角是一个小男孩和他的狗。

    <|assistant|>: 非常好的大纲！接下来，我们来确定一些关键情节和转折。你有任何具体的想法吗？例如，小男孩和他的狗在家附近的森林中发现一个神秘的地图。他们决定追寻地图的线索，这将带领他们穿越森林，遇到各种困难和挑战。最后，他们找到了一份宝藏，但他们意识到真正的宝藏其实是他们之间的友谊。你觉得如何？

    <|user|>: 可以的，不过可以加一些对于森林细节的描述。

    <|assistant|>: 明白了，我会以这种风格来生成故事。稍等一下，我马上生成故事：

    在一天阳光明媚的早晨，小男孩Tommy和他的狗Max在后院发现了一个被风吹来的神秘地图。地图上标记着一条通向森林深处的路线，一颗大大的"X"标记在地图的另一头。

    "Max，看看这个！我们找到宝藏地图了！" Tommy兴奋地对他的狗说。他们决定冒险寻找宝藏。他们沿着地图上的路线，穿过森林，爬过山丘，甚至跨过一条吓人的独木桥。

    他们手牵手跨过了这条河。这时候，突然冲出一只大熊，小男孩和狗狗合力战胜大熊。

    最终他们找到了宝藏，并在夕阳下走回了家。

    故事已经生成完成，你是否喜欢这个故事呢？

    <|user|>: 可以的，故事生成的不错，我很喜欢！

    <|assistant|>: 很高兴你喜欢！接下来，我们可以开始创建故事的插图。你更喜欢哪种风格的插图？是卡通画风格，还是赛博朋克风格

    <|user|>: 我更喜欢赛博朋克风格

    <|assistant|>: 好的，我将开始绘本创作，为上述内容分别配上插图

    正在生成第一段内容的配图：<|startofthink|>```JSON\n{\n   "api_name": "image_generation",\n    "parameters": {\n      "text": "在一天阳光明媚的早晨，小男孩Tommy和他的狗Max在后院发现了一个被风吹来的神秘地图。地图上标记着一条通向森林深处的路线，一颗大大的"X"标记在地图的另一头。", "idx": "0", "type": "赛博朋克"\n   }\n}\n```<|endofthink|>

    正在生成第二段内容的配图：<|startofthink|>```JSON\n{\n   "api_name": "image_generation",\n    "parameters": {\n      "text": "Max，看看这个！我们找到宝藏地图了！ Tommy兴奋地对他的狗说。他们决定冒险寻找宝藏。他们沿着地图上的路线，穿过森林，爬过山丘，甚至跨过一条吓人的独木桥。", "idx": "1",  "type": "赛博朋克"\n   }\n}\n```<|endofthink|>

    正在生成第三段内容的配图：<|startofthink|>```JSON\n{\n   "api_name": "image_generation",\n    "parameters": {\n      "text": "他们手牵手跨过了这条河。这时候，突然冲出一只大熊，小男孩和狗狗合力战胜大熊。", "idx": "2",  "type": "赛博朋克"\n   }\n}\n```<|endofthink|>

    正在生成第四段内容的配图：<|startofthink|>```JSON\n{\n   "api_name": "image_generation",\n    "parameters": {\n      "text": "最终他们找到了宝藏，并在夕阳下走回了家。", "idx": "3",  "type": "赛博朋克"\n   }\n}\n```<|endofthink|>

    已经为你生成了一部关于科学家木子的故事，每段内容都配有卡通画风格的插图。如果需要调整或有其他想法，请随时告诉我。
   【角色扮演要求】
    上面多轮角色对话是提供的创作一个绘本要和用户沟通的样例，请按照上述的询问步骤来引导用户完成绘本生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含user或者<|user|>的内容。
"""


#KEY_TEMPLATE = """（提醒：请参照上述的多轮对话历史的询问步骤，引导用户完成绘本生成，但不要生成多轮对话，回复不要包含<|user|>或者<|user|>的内容。）"""
KEY_TEMPLATE = ""

MAX_SCENE = 4

env_file = '.env'
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
your_modelscope_api_token = None
your_dashscope_api_token = None
os.environ['MODEL_CONFIG_FILE'] = 'cfg_model_template.json'
os.environ['TOOL_CONFIG_FILE'] = 'cfg_tool_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'
if your_modelscope_api_token is not None:
    os.environ['MODELSCOPE_API_TOKEN'] = your_modelscope_api_token
if your_dashscope_api_token is not None:
    os.environ['DASHSCOPE_API_KEY'] = your_dashscope_api_token

IMAGE_TEMPLATE_PATH = [
    'img_example/1.png',
    'img_example/2.png',
]


with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()

tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', None)
model_cfg_file = os.getenv('MODEL_CONFIG_FILE', None)
tool_cfg = Config.from_file(tool_cfg_file)
model_cfg = Config.from_file(model_cfg_file)

model_name = 'modelscope-agent-7b'
llm = LLMFactory.build_llm(model_name, model_cfg)


def init_agent(state):
# ----------agent 对象初始化--------------------
    prompt_generator = MSPromptGenerator(
        system_template=SYSTEM_PROMPT,
        instruction_template=INSTRUCTION_TEMPLATE)
    # tools 

    print_story_tool = PrintStoryTool()
    show_img_example_tool = ShowExampleTool(IMAGE_TEMPLATE_PATH)
    image_generation_tool = ImageGenerationTool(tool_cfg)

    additional_tool_list = {
        print_story_tool.name: print_story_tool,
        show_img_example_tool.name: show_img_example_tool,
        image_generation_tool.name: image_generation_tool
    }

    agent = AgentExecutor(
        llm,
        tool_cfg,
        prompt_generator=prompt_generator,
        tool_retrieval=False,
        additional_tool_list=additional_tool_list)

    agent.set_available_tools(additional_tool_list.keys())
    state['agent'] = agent


with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:
    state = gr.State({})
    demo.load(init_agent, inputs=[state], outputs=[])

    max_scene = MAX_SCENE

    with gr.Row():
        gr.HTML(
            """<h1 align="left" style="min-width:200px; margin-top:0;">StoryAgent</h1>"""
        )
        gr.HTML("""<div align="center">
            <div style="display:flex; gap: 0.25rem;" align="center">
                <a href='https://github.com/modelscope/modelscope-agent'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
                <a href="https://arxiv.org/abs/2309.00986"><img src="https://img.shields.io/badge/Arxiv-2304.14178-red"></a>
            </div>
        </div>""")
        status_display = gr.HTML(
            "", elem_id="status_display", visible=False, show_label=False)

    with gr.Row(elem_id="container_row").style(equal_height=True):

        with gr.Column(scale=6):
            
            gr.HTML(
            """<span data-testid="block-info" class="svelte-1gfkn6j">生成内容</span>"""
            )

            output_image = [None] * max_scene
            output_text = [None] * max_scene

            for i in range(0, max_scene, 2):
                with gr.Row():
                    with gr.Column():
                        output_image[i] = gr.Image(
                            label=f'示例图片{i + 1}',
                            interactive=False,
                            height=400,
                            visible=False,
                            show_progress=False)
                        output_text[i] = gr.Textbox(
                            label=f'故事情节{i + 1}', lines=2, interactive=False, visible=False, show_progress=False)
                    with gr.Column():
                        output_image[i + 1] = gr.Image(
                            label=f'示例图片{i +2}', interactive=False, height=400, visible=False, show_progress=False)
                        output_text[i + 1] = gr.Textbox(
                            label=f'故事情节{i + 2}', lines=2, interactive=False, visible=False, show_progress=False)

        with gr.Column(min_width=470, scale=6, elem_id='settings'):

            chatbot = ChatBot(
                elem_id="chatbot",
                elem_classes=["markdown-body"],
                show_label=False,
                value=[[None, PROMPT_START]],
                height=400)
            with gr.Row(elem_id="chat-bottom-container"):
                with gr.Column(min_width=70, scale=1):
                    clear_session_button = gr.Button(
                        "清除", elem_id='clear_session_button', default_value=True)
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="请输入你想要生成的故事情节吧～",
                        elem_id="chat-input").style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("发送", variant="primary")
                with gr.Column(min_width=110, scale=1):
                    regenerate_button = gr.Button(
                        "重新生成", elem_id='regenerate_button')

            gr.Examples(
                examples=['给我生成一个向日葵刺猬的故事', '增加一个它的一路坎坷', '可以的，故事生成的不错，我很喜欢！', '卡通画风格'],
                inputs=[user_input],
                examples_per_page=20,
                label="示例",
                elem_id="chat-examples")

    

    def story_agent(*inputs):

        

        max_scene = MAX_SCENE

        user_input = inputs[0]
        chatbot = inputs[1]
        state = inputs[2]
        output_component = list(inputs[3:])

        agent = state['agent']
    
        def reset_component():
            for i in range(max_scene):
                output_component[i] = gr.Image.update(visible=False)
                output_component[i+max_scene] = gr.Textbox.update(visible=False)

        # reset_component()

        chatbot.append((user_input, None))
        yield chatbot, *output_component
        
        def update_component(exec_result):
            exec_result = exec_result['result']
            name = exec_result.pop('name')
            if name ==  'print_story_tool':
                pass
                #output_component[0] = gr.Textbox.update(**exec_result)
            elif name == 'show_image_example':
                for i, r in enumerate(exec_result['result']):
                    output_component[i] = gr.Image.update(**r)
            elif name == 'image_generation':
                idx = int(exec_result.pop('idx'))
                output_component[idx] = gr.Image.update(**exec_result['img_result'])
                output_component[idx+max_scene] = gr.Textbox.update(**exec_result['text_result'])

        response = ''
        for frame in agent.stream_run(user_input+KEY_TEMPLATE, remote=True):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '') 
            print(frame)
            llm_result = llm_result.split("<|user|>")[0].strip()
            if len(exec_result) != 0:
                # llm_result
                update_component(exec_result)
                frame_text = ' '
            else:
                # action_exec_result
                frame_text = llm_result
                pattern = r"<\|startofthink\|>[\s\S]*<\|endofthink\|>"
                while re.search(pattern, frame_text):
                    frame_text = re.sub(pattern, "", frame_text)
            
            response = f'{response}\n{frame_text}'
            print(f'response: {response}')
            chatbot[-1] = (user_input, response)
            yield chatbot, *copy.deepcopy(output_component)
            # print ("response: ", response)
        
#         chatbot[-1] = (user_input, response)
#         yield chatbot, *output_component

    # ---------- 事件 ---------------------

    stream_predict_input = [user_input, chatbot, state, *output_image, *output_text]
    stream_predict_output = [chatbot, *output_image, *output_text]

    clean_outputs_start = ['', gr.update(value=[(None, PROMPT_START)])] + [None] * max_scene + [''] * max_scene
    clean_outputs = [''] + [None] * max_scene + [''] * max_scene
    clean_outputs_start_target = [user_input, chatbot, *output_image, *output_text]
    clean_outputs_target = [user_input, *output_image, *output_text]
    user_input.submit(
        story_agent,
        inputs=stream_predict_input,
        outputs=stream_predict_output,
        show_progress=True)
    user_input.submit(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    submitBtn.click(
        story_agent,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)
    submitBtn.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    regenerate_button.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    regenerate_button.click(
        story_agent,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)

    def clear_session(state):
        agent = state['agent']
        agent.reset()

    clear_session_button.click(fn=clear_session, inputs=[state], outputs=[])
    clear_session_button.click(
        fn=lambda: clean_outputs_start, inputs=[], outputs=clean_outputs_start_target)
  
    demo.title = "StoryAgent 🎁"
    
    demo.queue(concurrency_count=1, status_update_rate='auto', api_open=False)
    demo.launch(show_api=False, share=False)
