from .chatglm3_prompt import ChatGLMPromptGenerator
from .messages_prompt import MessagesGenerator
from .mrkl_prompt import MrklPromptGenerator
from .ms_prompt import MSPromptGenerator
from .prompt import PromptGenerator
from .qwen_prompt import QwenPromptGenerator
from .raw_prompt_builder import build_raw_prompt

prompt_generators = {
    'ChatGLMPromptGenerator': ChatGLMPromptGenerator,
    'MessagesGenerator': MessagesGenerator,
    'MrklPromptGenerator': MrklPromptGenerator,
    'MSPromptGenerator': MSPromptGenerator,
    'PromptGenerator': PromptGenerator,
    'QwenPromptGenerator': QwenPromptGenerator
}
