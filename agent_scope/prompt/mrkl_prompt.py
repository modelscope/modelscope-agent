import json

from .prompt import LengthConstraint, PromptGenerator

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

MRKL_DEFAULT_EXEC_TEMPLATE = """Observation: <exec_result>\n"""

TOOL_DESC = (
    '{name_for_model}: {name_for_human} API. {description_for_model} 输入参数: {parameters}'
)

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
                 llm=None,
                 length_constraint=LengthConstraint(),
                 **kwargs):
        super().__init__(
            system_template=system_template,
            instruction_template=instruction_template,
            user_template=user_template,
            exec_template=exec_template,
            assistant_template=assistant_template,
            sep=sep,
            llm=llm,
            length_constraint=length_constraint,
            **kwargs)

    def init_prompt(self, task, tool_list, knowledge_list, **kwargs):
        if len(self.history) == 0:
            super().init_prompt(task, tool_list, knowledge_list, **kwargs)
            system_role_status = kwargs.get('system_role_status', False)
            tool_names = [f'\'{str(tool.name)}\'' for tool in tool_list]
            tool_names = ','.join(tool_names)
            self.system_prompt = self.system_prompt.replace(
                '<tool_names>', tool_names)

            if system_role_status:
                system_message = {
                    'role': 'system',
                    'content': self.system_prompt
                }
                self.history.insert(0, system_message)
            else:
                self.history[0]['content'] = self.system_prompt + self.history[
                    0]['content']
        else:
            self.history.append({
                'role':
                'user',
                'content':
                self.user_template.replace('<user_input>', task)
            })
            self.history.append({
                'role': 'assistant',
                'content': self.assistant_template
            })

        return self.system_prompt

    def get_tool_str(self, tool_list):
        tool_texts = []
        for tool in tool_list:
            tool_texts.append(
                TOOL_DESC.format(
                    name_for_model=tool.name,
                    name_for_human=tool.name,
                    description_for_model=tool.description,
                    parameters=json.dumps(tool.parameters,
                                          ensure_ascii=False)))
            # + ' ' + FORMAT_DESC['json'])
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
            self.history[-1]['content'] += exec_result
        self.prompt = self.prompt_preprocessor(self.history)
        return self.prompt
