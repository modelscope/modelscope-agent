from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


class ImageChatTool(ModelscopePipelineTool):
    default_model = 'damo/multi-modal_mplug_owl_multimodal-dialogue_7b'
    description = '图文对话和图像描述服务，针对输入的图片和用户的文本输入，给出文本回复'
    name = 'image-chat'
    parameters: list = [{
        'name': 'image',
        'description': '用户输入的图片',
        'required': True
    }, {
        'name': 'text',
        'description': '用户输入的文本',
        'required': True
    }]
    task = Tasks.multimodal_dialogue

    def construct_image_chat_input(self, **kwargs):
        image = kwargs.pop('image', '')
        text = kwargs.pop('text', '')

        system_prompt_1 = 'The following is a conversation between a curious human and AI assistant.'
        system_prompt_2 = "The assistant gives helpful, detailed, and polite answers to the user's questions."
        messages = {
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt_1 + ' ' + system_prompt_2
                },
                {
                    'role': 'user',
                    'content': [{
                        'image': image
                    }]
                },
                {
                    'role': 'user',
                    'content': text
                },
            ]
        }
        return messages

    def _remote_parse_input(self, *args, **kwargs):
        messages = self.construct_image_chat_input(**kwargs)
        return {'input': messages}

    def _local_parse_input(self, *args, **kwargs):
        return (self.construct_image_chat_input(**kwargs)), {}
