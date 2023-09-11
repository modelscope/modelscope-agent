from __future__ import annotations
import os
import sys
sys.path.append('../../')
from functools import partial

import gradio as gr
from dotenv import load_dotenv
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator
from modelscope_agent.retrieve import ToolRetrieval
from predict import generate_story

from modelscope.utils.config import Config



SYSTEM_PROMPT = "<|system|>:你是Story Agent，是一个大语言模型，可以根据用户的输入自动生成相应的绘本。"

INSTRUCTION_TEMPLATE = """当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n
<|user|>: 生成一个老虎和狮子的故事，分成2段展示 

<|assistant|>: 从前，有一只勇敢的狮子和一只强大的老虎生活在同一个森林里。这两只野兽都非常自豪，并且它们之间一直存在着一种友好而竞争的关系。

一天，狮子和老虎相遇了，它们开始比试力气。狮子在速度和力量上都很强大，而老虎则凭借其敏捷和灵活性而闻名。它们决定进行一场友谊比赛，看看谁能在一系列挑战中获胜。

<|user|>: 请调用英文图片生成模型生成图片：从前，有一只勇敢的狮子和一只强大的老虎生活在同一个森林里。必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。

<|assistant|>: <|startofthink|>```JSON\n{\n   "api_name": "modelscope_image-generation",\n    "parameters": {\n      "text": "There was a brave lion and a powerful tiger living in the same forest"\n   }\n}\n```<|endofthink|>\n\n<|startofexec|>```JSON\n{\n   "result": "http://xdp-expriment.oss-cn-zhangjiakou.aliyuncs.com/yeqinghao.yqh/generated_images/6y09j0jmnkm1e6o3nfy7pc6e951az95n_1.jpg"\n}\n```<|endofexec|>\n![IMAGEGEN](http://xdp-expriment.oss-cn-zhangjiakou.aliyuncs.com/yeqinghao.yqh/toolformer_imagegen/generated_images/6y09j0jmnkm1e6o3nfy7pc6e951az95n_1.jpg)

<|user|>: 请调用英文图片生成模型生成图片：一天，狮子和老虎相遇了，它们开始比试力气。必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。

<|assistant|>: <|startofthink|>```JSON\n{\n   "api_name": "modelscope_image-generation",\n    "parameters": {\n      "text": "The lion and the tiger encountered each other and decided to test their strength"\n   }\n}\n```<|endofthink|>\n\n<|startofexec|>```JSON\n{\n   "result": "https://xdp-expriment.oss-cn-zhangjiakou.aliyuncs.com/modelscope%2Fimage%2Fc63990c47fd34508.jpg"\n}\n```<|endofexec|>\n![IMAGEGEN](https://xdp-expriment.oss-cn-zhangjiakou.aliyuncs.com/modelscope%2Fimage%2Fc63990c47fd34508.jpg)
"""

MAX_SCENE = 4

# sys.path.append('../../')
load_dotenv('../../config/.env', override=True)

os.environ['TOOL_CONFIG_FILE'] = '../../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../../config/cfg_model_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'
os.environ['MODELSCOPE_API_TOKEN'] = 'xxx'
os.environ['DASHSCOPE_API_KEY'] = 'xxx'
os.environ['OPENAI_API_KEY'] = 'xxx'

with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()

with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:

    max_scene = MAX_SCENE

    # agent 对象初始化

    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

    tool_cfg = Config.from_file(tool_cfg_file)
    # model_name = 'ms_gpt'
    model_cfg = Config.from_file(model_cfg_file)

    model_name = 'openai'
    # model_cfg = {
    #     'modelscope-agent-qwen-7b': {
    #         'model_id': 'damo/MSAgent-Qwen-7B',
    #         'model_revision': 'v1.0.2',
    #         'use_raw_generation_config': True,
    #         'custom_chat': True
    #     }
    # }

    retrieve = ToolRetrieval(top_k=3)
    prompt_generator = MSPromptGenerator(
        system_template=SYSTEM_PROMPT,
        instruction_template=INSTRUCTION_TEMPLATE)

    llm = LLMFactory.build_llm(model_name, model_cfg)
    agent = AgentExecutor(
        llm,
        tool_cfg,
        prompt_generator=prompt_generator,
        tool_retrieval=retrieve)

    generate_story_p = partial(
        generate_story, max_scene=max_scene, agent=agent)

    with gr.Row():
        gr.HTML(
            """<h1 align="left" style="min-width:200px; margin-top:0;">ModelScopeGPT</h1>"""
        )
        status_display = gr.HTML(
            "", elem_id="status_display", visible=False, show_label=False)

    with gr.Row(elem_id="container_row").style(equal_height=True):

        with gr.Column(
                scale=8,
                elem_classes=["chatInterface", "chatDialog", "chatContent"]):
            # with gr.Column(elem_id="chat-container"):
            output_image = [None] * max_scene
            output_text = [None] * max_scene

            for i in range(0, max_scene, 2):
                with gr.Row():
                    with gr.Column():
                        output_image[i] = gr.Image(
                            label=f'绘本图片{i + 1}',
                            interactive=False,
                            height=200)
                        output_text[i] = gr.Textbox(
                            label=f'故事情节{i + 1}', lines=4, interactive=False)
                    with gr.Column():
                        output_image[i + 1] = gr.Image(
                            label=f'绘本图片{i +2}', interactive=False, height=200)
                        output_text[i + 1] = gr.Textbox(
                            label=f'故事情节{i + 2}', lines=4, interactive=False)

        with gr.Column(min_width=470, scale=4, elem_id='settings'):
            gr.HTML("""
                <div class="robot-info">
                    <img src="https://img.alicdn.com/imgextra/i4/\
                    O1CN01kpkVcX1wSCO362MH4_!!6000000006306-1-tps-805-805.gif"></img>
                    <div class="robot-info-text">
                        我是story agent。
                    </div>
                </div>
            """)

            with gr.Row(elem_id="chat-bottom-container"):
                with gr.Column(min_width=70, scale=1):
                    clear_session_button = gr.Button(
                        "清除", elem_id='clear_session_button')
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
                examples=[
                    '嗨，storyagent，我正在为一个新的电子绘本构思一个故事。我希望这是一个关于友谊和冒险的故事，主角是一只勇敢的小狐狸和其他小动物，分成三幕来生成。',
                    '主角是一只勇敢的小狐狸和其他小动物，分成2幕来生成。', '主角是一只勇敢的小狐狸和其他小动物，',
                    '嗨，storyagent，我正在为一个新的电子绘本构思一个故事。我希望这是一个关于友谊和冒险的故事，主角是一只勇敢的小狐狸和其他小动物。'
                ],
                inputs=[user_input],
                examples_per_page=20,
                label="示例",
                elem_id="chat-examples")

            steps = gr.Slider(
                minimum=1,
                maximum=max_scene,
                value=1,
                step=1,
                label='生成绘本的数目',
                interactive=True)

    stream_predict_input = [user_input, steps]
    stream_predict_output = [*output_image, *output_text]

    clean_outputs = [''] + [None] * max_scene + [''] * max_scene
    clean_outputs_target = [user_input, *output_image, *output_text]

    user_input.submit(
        generate_story_p,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)
    user_input.submit(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    submitBtn.click(
        generate_story_p,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)
    submitBtn.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    regenerate_button.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    regenerate_button.click(
        generate_story_p,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)

    def clear_session():
        agent.reset()

    clear_session_button.click(fn=clear_session, inputs=[], outputs=[])
    clear_session_button.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    demo.title = "StoryAgent 🎁"
    demo.queue(concurrency_count=10, status_update_rate='auto', api_open=False)
    demo.launch(show_api=False, share=True)
