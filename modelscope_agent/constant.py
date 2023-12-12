DEFAULT_MODEL_CONFIG = {
    'qwen': {
        'en': {
            'prompt_generator': 'MrklPromptGenerator',
            'action_parser': 'MRKLActionParser'
        },
        'zh': {
            'prompt_generator': 'MrklPromptGenerator',
            'action_parser': 'MRKLActionParser'
        }
    },
    'qwen_plus': {
        'prompt_generator': 'MessagesGenerator',
        'action_parser': 'MRKLActionParser'
    },
    'chatglm': {
        'prompt_generator': 'ChatGLMPromptGenerator',
        'action_parser': 'ChatGLMActionParser'
    },
    'gpt': {
        'prompt_generator': 'MrklPromptGenerator',
        'action_parser': 'MRKLActionParser'
    },
    'openai': {
        'prompt_generator': 'MrklPromptGenerator',
        'action_parser': 'MRKLActionParser'
    }
}
