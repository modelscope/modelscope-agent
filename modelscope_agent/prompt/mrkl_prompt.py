import json

from .prompt import PromptGenerator

MRKL_DEFAULT_SYSTEM_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools: `

<tool_list>"""

MRKL_DEFAULT_INSTRUCTION_TEMPLATE = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [<tool_names>]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

MRKL_DEFAULT_USER_TEMPLATE = """Question: <user_input>\n"""

MRKL_DEFAULT_EXEC_TEMPLATE = """Observation: <exec_result>\nThought:"""

TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human}'
    + ' API. What is the {name_for_human} API useful for?'
    + ' {description_for_model} Parameters: {parameters}')

FORMAT_DESC = {
    'json':
    'Format the arguments as a JSON object.',
    'code':
    'Enclose the code within triple backticks (`)'
    + ' at the beginning and end of the code.'
}


class MrklPromptGenerator(PromptGenerator):

    def __init__(self,
                 system_template=MRKL_DEFAULT_SYSTEM_TEMPLATE,
                 instruction_template=MRKL_DEFAULT_INSTRUCTION_TEMPLATE,
                 user_template=MRKL_DEFAULT_USER_TEMPLATE,
                 exec_template=MRKL_DEFAULT_EXEC_TEMPLATE,
                 assistant_template='',
                 sep='\n\n',
                 prompt_max_length=10000):
        super().__init__(system_template, instruction_template, user_template,
                         exec_template, assistant_template, sep,
                         prompt_max_length)

    def init_prompt(self, task, tool_list, knowledge_list, llm_model,
                    **kwargs):
        super().init_prompt(task, tool_list, knowledge_list, llm_model,
                            **kwargs)
        tool_names = [f'\'{str(tool.name)}\'' for tool in tool_list]
        tool_names = ','.join(tool_names)
        self.system_prompt = self.system_prompt.replace(
            '<tool_names>', tool_names)
        return self.system_prompt

    def get_tool_str(self, tool_list):
        tool_texts = []
        for tool in tool_list:
            tool_texts.append(
                TOOL_DESC.format(
                    name_for_model=tool.name,
                    name_for_human=tool.name,
                    description_for_model=tool.description,
                    parameters=json.dumps(tool.parameters, ensure_ascii=False))
                + ' ' + FORMAT_DESC['json'])
        tool_str = '\n\n'.join(tool_texts)
        return tool_str

    def _generate(self, llm_result, exec_result: str):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """
        if len(llm_result) != 0:
            self.history[-1]['content'] += f'{llm_result}'
        if len(exec_result) != 0:
            exec_result = self.exec_template.replace('<exec_result>',
                                                     str(exec_result))
            function_content = {
                'role': 'function',
                'content': exec_result,
            }
            self.history.append(function_content)
        self.prompt = self.prompt_preprocessor(self.history,
                                               self.system_prompt)
        return self.prompt
