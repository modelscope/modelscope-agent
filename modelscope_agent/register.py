import functools

from modelscope_agent.action_parser import action_parsers
from modelscope_agent.prompt import prompt_generators


class Register:
    registered = dict()

    def __init__(self, init_classes):
        self.registered = init_classes

    def __call__(self, new_classes):
        self.registered.update(new_classes)


prompt_generator_register = Register(prompt_generators)
action_parser_register = Register(action_parsers)
