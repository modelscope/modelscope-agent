import copy
from typing import Union

from .raw_prompt_builder import build_raw_prompt

KNOWLEDGE_RESULT_PROMPT = 'Web search results: '


class PromptGenerator:

    def __init__(self,
                 system_template: str = '',
                 instruction_template: str = '',
                 user_template: str = '<user_input>',
                 exec_template: str = '',
                 assistant_template: str = '',
                 sep='\n\n',
                 prompt_max_length: int = 10000):
        """
        prompt genertor
        Args:
            system_template (str, optional): System template, normally the role of LLM.
            instruction_template (str, optional): Indicate the instruction for LLM.
            user_template (str, optional): Prefix before user input. Defaults to ''.
            exec_template (str, optional): A wrapper str for exec result.
            assistant_template (str, optional): Prefix before assistant response.
            Some LLM need to manully concat this prefix before generation.
            prompt_max_length (int, optional): max length of prompt. Defaults to 2799.

        """

        self.system_template = system_template
        self.instruction_template = instruction_template
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.exec_template = exec_template
        self.sep = sep

        self.prompt_max_length = prompt_max_length
        self.reset()

    def reset(self):
        self.prompt = ''
        self.history = []
        self.messages = []

    def init_prompt(self, task, tool_list, knowledge_list, llm_model,
                    **kwargs):
        """
        in this function, the prompt will be initialized.
        """
        self.prompt_preprocessor = build_raw_prompt(llm_model)

        prompt = self.sep.join(
            [self.system_template, self.instruction_template])
        prompt += '<knowledge><history>'

        knowledge_str = self.get_knowledge_str(knowledge_list)

        # knowledge
        prompt = prompt.replace('<knowledge>', knowledge_str)

        # get tool description str
        tool_str = self.get_tool_str(tool_list)
        prompt = prompt.replace('<tool_list>', tool_str)

        history_str = self.get_history_str()

        prompt = prompt.replace('<history>', history_str)

        self.system_prompt = copy.deepcopy(prompt)

        # user input
        user_input = self.user_template.replace('<user_input>', task)
        prompt += f'{self.sep}{user_input}'

        # assistant input
        prompt += f'{self.sep}{self.assistant_template}'

        # store history
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({
            'role': 'assistant',
            'content': self.assistant_template
        })

        self.prompt = prompt

        self.function_calls = self.get_function_list(tool_list)

        return self.prompt

    # TODO change the output from single prompt to artifacts including prompt, messages, funciton_call
    def generate(self, llm_result, exec_result: Union[str, dict]):
        if isinstance(exec_result, dict):
            exec_result = str(exec_result['result'])
        return self._generate(llm_result, exec_result)

    def _generate(self, llm_result, exec_result: str):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """
        if len(llm_result) != 0:
            self.prompt = f'{self.prompt}{llm_result}'
            self.history[-1]['content'] += f'{llm_result}'
        if len(exec_result) != 0:
            exec_result = self.exec_template.replace('<exec_result>',
                                                     str(exec_result))
            self.prompt = f'{self.prompt}{self.sep}{exec_result}'
            self.history[-1]['content'] += f'{self.sep}{exec_result}'

        return self.prompt

    # TODO: add Union[Text, Message] type for llm_result,
    #  add ExecResult = Text type for exec_result
    #  output would be a Union[Text, Messages]
    # In this case llm_result is Message, and exec_result is Function_call
    def _generate_messages(self, llm_result, exec_result: str):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """

        # init task should be
        if llm_result == '' and exec_result == '':
            return self.history

        # make sure set content  ''  not null
        function_call = llm_result.get('function_call', None)
        if function_call is not None:
            llm_result['content'] = ''
        self.history.append(llm_result)

        if exec_result is not None and function_call is not None:
            exec_message = {
                'role': 'function',
                'name': 'execute',
                'content': exec_result,
            }
            self.history.append(exec_message)

        return self.history

    def get_tool_str(self, tool_list):
        """generate tool list string

        Args:
            tool_list (List[str]): list of tools

        """

        tool_str = self.sep.join(
            [f'{i+1}. {t}' for i, t in enumerate(tool_list)])
        return tool_str

    # TODO move parse_tools_to_function from agent to here later
    def get_function_list(self, tool_list):
        """generate funciton call list from tools list

        Args:
            tool_list (List[str]): list of tools

        """
        functions = [tool.get_function() for tool in tool_list]
        return functions

    def get_knowledge_str(self, knowledge_list):
        """generate knowledge string

        Args:
            knowledge_list (List[str]): list of knowledges

        """

        knowledge = self.sep.join(
            [f'{i+1}. {k}' for i, k in enumerate(knowledge_list)])
        knowledge_str = f'{self.sep}{KNOWLEDGE_RESULT_PROMPT}{self.sep}{knowledge}' if len(
            knowledge_list) > 0 else ''
        return knowledge_str

    def get_history_str(self):
        """generate history string

        """
        history_str = ''
        for i in range(len(self.history)):
            history_item = self.history[len(self.history) - i - 1]
            text = history_item['content']
            if len(history_str) + len(text) + len(
                    self.prompt) > self.prompt_max_length:
                break
            history_str = f'{self.sep}{text.strip()}{history_str}'

        return history_str
