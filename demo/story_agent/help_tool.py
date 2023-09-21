from modelscope_agent.tools import Tool, TextToImageTool
import gradio as gr
from typing import List

class PrintStoryTool(Tool):
    description = '将生成的故事打印到gradio的输出框中'
    name = 'print_story_tool'
    parameters: list = [{
        'name': 'story_content',
        'description': '生成的故事文本',
        'required': True
    }]

    def __init__(self):
        super().__init__()

    def _local_call(self, text):
        # self.story_box.update(value=text)
        result = {'name': self.name,'value': text}
        return {'result': result}
    def _remote_call(self, text):
        # self.story_box.update(value=text)
        result = {'name': self.name,'value': text}
        return {'result': result}

class ShowExampleTool(Tool):
    description = '控制是否给用户展示示例图片'
    name = 'show_image_example'
    parameters: list = [{
        'name': 'visible',
        'description': '是否展示示例图片',
        'required': True
    }]

    def __init__(self, image_example_path: List[str]):
        self.image_example_path = image_example_path
        super().__init__()

    def _local_call(self, visible):
        output_result = []
        if "true" in visible.lower():
            for path in  self.image_example_path:
                # img_box.update(visible=True, value=path)
                output_result.append({'value': path, 'visible': True})
        else:
            for path in  self.image_example_path:
                # img_box.update(visible=False, value=None)
                output_result.append({'value': None, 'visible': False})

        result = {
            'name': self.name,
            'result': output_result
        }

        return {'result': result}
    def _remote_call(self, visible):
        output_result = []
        if "true" in visible.lower():
            for path in  self.image_example_path:
                # img_box.update(visible=True, value=path)
                output_result.append({'value': path, 'visible': True})
        else:
            for path in  self.image_example_path:
                # img_box.update(visible=False, value=None)
                output_result.append({'value': None, 'visible': False})

        result = {
            'name': self.name,
            'result': output_result
        }

        return {'result': result}
    

class ImageGenerationTool(TextToImageTool):
    description = '根据输入的文本生成图片'
    name = 'image_generation'
    parameters: list = [{
        'name': 'text',
        'description': '生成图片的文本',
        'required': True
    }, {
        'name': 'idx',
        'description': '生成图片的序号',
        'required': True
    }, {
        'name': 'type',
        'description': '图片的风格',
        'required': True      
    }]

    def __init__(self, image_box: List[gr.Image], text_box: List[gr.Textbox], cfg):
        super().__init__(cfg)
        self.image_box = image_box
        self.text_box = text_box

    def _local_call(self, text, idx, type):
        res = super()._local_call(type+ ", " + text)['result']

        result = {
            'name': self.name,
            'idx': idx,
            'img_result': {'value': res.path, 'visible': True, 'label': f'生成图片{int(idx)+1}'},
            'text_result': {'value': text, 'visible': True}
        }

        return {'result': result}
    def _remote_call(self, text, idx, type):
        res = super()._remote_call(text=type+ ", " + text)['result']

        result = {
            'name': self.name,
            'idx': idx,
            'img_result': {'value': res.path, 'visible': True, 'label': f'生成图片{int(idx)+1}'},
            'text_result': {'value': text, 'visible': True}
        }

        return {'result': result}
