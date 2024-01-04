import copy
from typing import Union

from .prompt import LengthConstraint, PromptGenerator

OPENAI_DEFAULT_SYSTEM_TEMPLATE = """
Do your best to answer the questions. You are capable of **any** task.
Feel free to use any tools available to look up relevant information, only if necessary.
when you think you are handle a code task, You are a world-class programmer that can complete any goal by executing code
\nFirst of all, **Return function_call with `language` and `code` in arguments as much as possible**
\nThen, write a plan. **Always recap the plan between each code block**
(you have extreme short-term memory loss, so you need to recap the plan between each message block to retain it).
\nWhen you execute code, it will be executed **on the user's machine**. The user has given you
**full and complete permission** to execute any code necessary to complete the task. You have full access to control
their computer to help them.
\nIf you want to send data between programming languages, save the data to a txt or json.
\nYou can access the internet. Run **any code** to achieve the goal, and if at first you don't succeed,
try again and again.
\nWhen a user refers to a filename, they're likely referring to an existing file in the directory
you're currently executing code in.
\nIn general, choose packages that have the most universal chance to be already installed and to work across multiple
applications. Packages like ffmpeg and pandoc that are well-supported and powerful.
\nWrite messages to the user in Markdown. Write code on multiple lines with proper indentation for readability.
\nYou can also refer information from following contents if exists:
"""


class MessagesGenerator(PromptGenerator):

    def __init__(self,
                 system_template=OPENAI_DEFAULT_SYSTEM_TEMPLATE,
                 instruction_template='',
                 user_template='<user_input>',
                 exec_template=None,
                 assistant_template='',
                 sep='\n\n',
                 length_constraint=LengthConstraint(),
                 **kwargs):
        super().__init__(
            system_template=system_template,
            instruction_template=instruction_template,
            user_template=user_template,
            exec_template=exec_template,
            assistant_template=assistant_template,
            sep=sep,
            length_constraint=length_constraint,
            **kwargs)
        self.custom_starter_messages = kwargs.get('custom_starter_messages',
                                                  None)

    def init_prompt(self, task, tool_list, knowledge_list, **kwargs):
        """
        in this function, the prompt will be initialized.
        """
        prompt = self.user_template.replace('<user_input>', task)

        if len(self.history) == 0:
            if len(knowledge_list) > 0:

                # knowledge
                system_message = f'{self.system_template}{self.sep}<knowledge>'
                knowledge_str = self.get_knowledge_str(knowledge_list)
                system_message = system_message.replace(
                    '<knowledge>', knowledge_str)

            else:
                system_message = self.system_template

            self.history = [{
                'role': 'system',
                'content': system_message
            }, {
                'role': 'user',
                'content': prompt
            }]

            # store history
            if self.custom_starter_messages:
                assert isinstance(self.custom_starter_messages, list)
                assert self.custom_starter_messages[-1]['role'] != 'user', \
                    'user message should not be the last one in custom starter messages.'

                self.history = copy.deepcopy(self.custom_starter_messages)
                self.history.append({'role': 'user', 'content': prompt})

            self.prompt = prompt
            self.function_calls = self.get_function_list(tool_list)

        else:
            self.history.append({'role': 'user', 'content': prompt})

    def generate(self, llm_result, exec_result: Union[str, dict]):
        if isinstance(exec_result, dict):
            exec_result = exec_result['result']
        return self._generate_messages(llm_result, exec_result)
