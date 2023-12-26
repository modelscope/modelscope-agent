from modelscope_agent.action_parser import (ChatGLMActionParser,
                                            MRKLActionParser, MsActionParser,
                                            OpenAiFunctionsActionParser)
from modelscope_agent.prompt import (ChatGLMPromptGenerator, MessagesGenerator,
                                     MrklPromptGenerator, MSPromptGenerator)

DEFAULT_MODEL_CONFIG = {
    'qwen': {
        'en': {
            'prompt_generator': MrklPromptGenerator,
            'action_parser': MRKLActionParser
        },
        'zh': {
            'prompt_generator': MrklPromptGenerator,
            'action_parser': MRKLActionParser
        }
    },
    'qwen-plus': {
        'prompt_generator': MessagesGenerator,
        'action_parser': MRKLActionParser
    },
    'chatglm': {
        'prompt_generator': ChatGLMPromptGenerator,
        'action_parser': ChatGLMActionParser
    },
    'gpt': {
        'prompt_generator': MrklPromptGenerator,
        'action_parser': MRKLActionParser
    },
    'openai': {
        'prompt_generator': MrklPromptGenerator,
        'action_parser': MRKLActionParser
    }
}
