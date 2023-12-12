DEFAULT_MODEL_CONFIG = {
    'qwen': {
        'en': {
            'prompt_generator': 'CustomPromptGenerator',
            'action_parser': 'MRKLActionParser'
        },
        'zh': {
            'prompt_generator': 'ZhCustomPromptGenerator',
            'action_parser': 'MRKLActionParser'
        }
    },
    'chatglm': {
        'prompt_generator': 'ChatGLMPromptGenerator',
        'action_parser': 'ChatGLMActionParser'
    },
    'gpt': {}
}
