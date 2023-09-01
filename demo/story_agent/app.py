from __future__ import annotations
import os
import sys
from functools import partial

import gradio as gr
from dotenv import load_dotenv
from gradio_chatbot import ChatBot
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm.ms_gpt import ModelScopeGPT
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator
from modelscope_agent.retrieve import ToolRetrieval
from predict import generate_story, stream_predict

from modelscope.utils.config import Config

sys.path.append('../../')
load_dotenv('config/.env', override=True)

SYSTEM_PROMPT = "<|system|>:你是Story Agent，是一个大语言模型，可以根据用户的输入自动生成相应的绘本。"

INSTRUCTION_TEMPLATE = """当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
然后你需要根据插件API调用结果生成合理的答复； 若需要生成故事情节，请在每一幕的开始用1, 2, 3...标记。\n\n<tool_list>"""

with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()

with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:

    max_scene = 4

    history = gr.State([])
    debug_info = gr.State([])

    # agent 对象

    model_cfg_file = os.getenv('LLM_CONFIG_FILE')
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')

    model_cfg = Config.from_file(model_cfg_file)
    tool_cfg = Config.from_file(tool_cfg_file)

    retrieve = ToolRetrieval(top_k=1)
    prompt_generator = MSPromptGenerator(
        system_template=SYSTEM_PROMPT,
        instruction_template=INSTRUCTION_TEMPLATE)

    llm = ModelScopeGPT(model_cfg)
    agent = AgentExecutor(llm, tool_cfg, tool_retrieval=retrieve)

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
            with gr.Row(elem_id="chat-container"):
                chatbot = ChatBot(
                    elem_id="chatbot",
                    elem_classes=["markdown-body"],
                    show_label=False)
                chatbot_classic = gr.Textbox(
                    lines=20,
                    visible=False,
                    interactive=False,
                    label='classic_chatbot',
                    elem_id='chatbot_classic')
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

        with gr.Column(min_width=470, scale=4, elem_id='settings'):
            gr.HTML("""
                <div class="robot-info">
                    <img src="https://img.alicdn.com/\
                    imgextra/i4/O1CN01kpkVcX1wSCO362MH4_!!6000000006306-1-tps-805-805.gif"></img>
                    <div class="robot-info-text">
                        我是story agent。
                    </div>
                </div>
            """)

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

            output_image = [None] * max_scene

            for i in range(0, max_scene, 2):
                with gr.Row():
                    output_image[i] = gr.Image(
                        label=f'绘本图片{i}', interactive=False, height=200)
                    output_image[i + 1] = gr.Image(
                        label=f'绘本图片{i+1}', interactive=False, height=200)

    stream_predict_input = [chatbot, user_input, steps]
    stream_predict_output = [chatbot, *output_image, status_display]

    clean_outputs = [''] + [None] * max_scene
    clean_outputs_target = [user_input, *output_image]

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
        return {
            chatbot: gr.update(value=[]),
            history: [],
            debug_info: [],
        }

    clear_session_button.click(
        fn=clear_session, inputs=[], outputs=[chatbot, history, debug_info])
    clear_session_button.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    demo.title = "StoryAgent 🎁"
    demo.queue(concurrency_count=10, status_update_rate='auto', api_open=False)
    demo.launch(show_api=False, share=True)
