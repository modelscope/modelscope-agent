from __future__ import annotations
import sys
sys.path.append('../../')
from dotenv import load_dotenv
load_dotenv('../../config/.env', override=True)
import gradio as gr
from gradio_chatbot import ChatBot
from predict import stream_predict, upload_image
from modelscope_agent.llm import LLMFactory
from modelscope_agent.agent import AgentExecutor
import os
from modelscope.utils.config import Config
from functools import partial


os.environ['TOOL_CONFIG_FILE'] = '../../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../../config/cfg_model_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'

with open(os.path.join(os.path.dirname(__file__), 'main.css'), "r", encoding="utf-8") as f:
        MAIN_CSS_CODE = f.read()

with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:

    upload_image_url = gr.State("")

    # agent 对象
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

    model_cfg = Config.from_file(model_cfg_file)
    tool_cfg = Config.from_file(tool_cfg_file)

    model_name = 'modelscope-agent'

    llm = LLMFactory.build_llm(model_name, model_cfg)
    agent = AgentExecutor(llm, tool_cfg)

    stream_predict_p = partial(stream_predict, agent=agent)

    with gr.Row():
        gr.HTML("""<h1 align="left" style="min-width:200px; margin-top:0;">ModelScopeGPT</h1>""")
        status_display = gr.HTML("", elem_id="status_display", visible=False, show_label=False)

    with gr.Row(elem_id="container_row").style(equal_height=True):

        with gr.Column(scale=8, elem_classes=["chatInterface", "chatDialog", "chatContent"]):
            with gr.Row(elem_id="chat-container"):
                chatbot = ChatBot(elem_id="chatbot", elem_classes=["markdown-body"], show_label=False)
                chatbot_classic = gr.Textbox(lines=20, visible=False, interactive=False, label='classic_chatbot',
                                             elem_id='chatbot_classic')
            with gr.Row(elem_id="chat-bottom-container"):
                with gr.Column(min_width=70, scale=1):
                    clear_session_button = gr.Button("清除", elem_id='clear_session_button') 
                with gr.Column(min_width=100, scale=1):
                    upload_button = gr.UploadButton("上传图片", file_types=['image'])
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="和我聊聊吧～", elem_id="chat-input"
                    ).style(container=False)
                    uploaded_image_box = gr.HTML("", visible=False, show_label=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("发送", variant="primary")
                with gr.Column(min_width=110, scale=1):
                    regenerate_button = gr.Button("重新生成", elem_id='regenerate_button')
                
        with gr.Column(min_width=470, scale=4, elem_id='settings'):
            gr.HTML("""
                <div class="robot-info">
                    <img src="https://img.alicdn.com/imgextra/i4/O1CN01kpkVcX1wSCO362MH4_!!6000000006306-1-tps-805-805.gif"></img>
                    <div class="robot-info-text">
                        我是ModelScopeGPT（魔搭GPT）， 是一个大小模型协同的agent系统。我具备多种能力，可以通过大模型做中枢（controller），来控制魔搭社区的各种多模态模型api回复用户的问题。除此之外，我还集成了知识库检索引擎，可以解答用户在魔搭社区使用模型遇到的问题以及模型知识相关问答。
                    </div>
                </div>
            """)

            gr.Examples(examples=[
                '写一个 2023 上海世界人工智能大会 20 字以内的口号，并念出来',
                '生成一个有山有水的图',
                '生成一段描述两个小狗玩耍的视频',
                '生成个20字描述新出的vision pro VR眼镜的文案，女声朗读，并转成视频',
                '写一首简短的夏天落日的诗',
                '语音播放',
                '生成个图片看看',
                '你可以扮演一位历史人物，假如你是孔子，你觉得自己最得意写的哪本书？语音回复',
                '那你的众多弟子中你最喜欢谁，继续语音回复我',
                '按照给定的schema抽取出下面文本对应的信息，schema：{"人物": null, "地理位置": null, "组织机构": null, "时间": null}\n2019年，中国科学院大学在北京举行了第六届“未来之星”论坛，来自全球的200多名青年科学家参加了此次论坛，包括李四，王五等。',
                '从下面的地址，找到省市区等元素，地址：浙江杭州市江干区九堡镇三村村一区',
                'ChatPLUG模型怎么使用，给个代码',
                'ChatPLUG模型链接发我下',
                '有没有支持开放域对话的模型',
                '魔搭社区如何接入一个新模型',
                #'这个模型有什么特点',
                #'有文档吗，链接发我下',
                #'怎么联系你们呢'
            ], inputs=[user_input], examples_per_page=20, label="示例", elem_id="chat-examples")

    stream_predict_input = [chatbot, user_input, upload_image_url]
    stream_predict_output = [chatbot, status_display]

    clean_outputs = [gr.update(value=''), '', '']
    clean_outputs_target = [user_input, uploaded_image_box, upload_image_url]

    user_input.submit(stream_predict_p,
                      stream_predict_input,
                      stream_predict_output,
                      show_progress=True)
    user_input.submit(fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

    submitBtn.click(stream_predict_p,
                    stream_predict_input,
                    stream_predict_output,
                    show_progress=True)
    submitBtn.click(fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)


    regenerate_button.click(fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    regenerate_button.click(stream_predict_p,
                            stream_predict_input,
                            stream_predict_output,
                            show_progress=True)
    
    upload_button.upload(upload_image, upload_button, [uploaded_image_box, upload_image_url])

    def clear_session():
        agent.reset()
        return {
            chatbot: gr.update(value=[]),
            uploaded_image_box: '',
            upload_image_url: '',
        }

    clear_session_button.click(fn=clear_session, inputs=[], outputs=[chatbot, uploaded_image_box, upload_image_url])

    demo.title = "ModelScopeGPT 🎁"
    demo.queue(concurrency_count=10, status_update_rate='auto', api_open=False)
    demo.launch(show_api=False, share=True)
