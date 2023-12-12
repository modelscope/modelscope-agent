import functools


class Register:
    registered = []

    def __call__(self, cls):
        self.register(cls)
        return cls

    def register(self, cls):
        self.registered.append(cls)


prompt_generator_register = Register()
action_parser_register = Register()
