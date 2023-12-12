
DEFAULT_MODEL_CONFIG = {
    "qwen": {
        "en": {
            "prompt_generator": "CustomPromptGenerator",
            "action_parser": "MessagesActionParser"
        },
        "zh": {
            "prompt_generator": "ZhCustomPromptGenerator",
            "action_parser": "MessagesActionParser"
        }
    },
    "chatglm": {
        "prompt_generator": "ChatGLMPromptGenerator",
        "action_parser": "ChatGLMActionParser"
    },
    "gpt": {}
}








