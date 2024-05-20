from typing import Union

from modelscope_agent.tools.base import register_tool

from modelscope.utils.constant import Tasks
from .pipeline_tool import ModelscopePipelineTool


@register_tool('image-chat')
class ImageChatTool(ModelscopePipelineTool):
    default_model = 'damo/multi-modal_mplug_owl_multimodal-dialogue_7b'
    description = '图文对话和图像描述服务，针对输入的图片和用户的文本输入，给出文本回复'
    name = 'image-chat'
    parameters: list = [{
        'name': 'image',
        'description': '用户输入的图片',
        'required': True,
        'type': 'string'
    }, {
        'name': 'text',
        'description': '用户输入的文本',
        'required': True,
        'type': 'string'
    }]
    task = Tasks.multimodal_dialogue
    url = 'https://api-inference.modelscope.cn/api-inference/v1/models/damo/multi-modal_mplug_owl_multimodal-dialogue_7b'  # noqa E501

    def call(self, params: str, **kwargs) -> str:
        result = super().call(params, **kwargs)
        image_chat = result['Data']['text']
        return image_chat

    def _verify_args(self, params: str) -> Union[str, dict]:
        params = super()._verify_args(params)
        image = params.pop('image', '')
        text = params.pop('text', '')
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
        params = {'input': messages}
        return params
