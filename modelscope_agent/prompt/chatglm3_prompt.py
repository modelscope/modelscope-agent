import json
from .prompt import PromptGenerator

CHATGLM_DEFAULT_SYSTEM_TEMPLATE = """<|system|>\nAnswer the following questions as best you can. You have access to the following tools:"""

CHATGLM_DEFAULT_INSTRUCTION_TEMPLATE = "<tool_list>"

CHATGLM_DEFAULT_USER_TEMPLATE = """<|user|>\n<user_input>"""

CHATGLM_DEFAULT_EXEC_TEMPLATE = """<|observation|>\n<exec_result>"""

CHATGLM_DEFAULT_ASSISTANT_TEMPLATE = """<|assistant|>"""

class ChatGLMPromptGenerator(PromptGenerator):

    def __init__(self,
                 system_template=CHATGLM_DEFAULT_SYSTEM_TEMPLATE,
                 instruction_template=CHATGLM_DEFAULT_INSTRUCTION_TEMPLATE,
                 user_template=CHATGLM_DEFAULT_USER_TEMPLATE,
                 exec_template=CHATGLM_DEFAULT_EXEC_TEMPLATE,
                 assistant_template=CHATGLM_DEFAULT_ASSISTANT_TEMPLATE,
                 sep='\n',
                 prompt_max_length=10000):
        super().__init__(system_template, instruction_template, user_template,
                         exec_template, assistant_template, sep,
                         prompt_max_length)

    def get_tool_str(self, tool_list):
        tool_json = json.loads('[' + ','.join([str(item) for item in tool_list]) + ']')
        return json.dumps(tool_json, ensure_ascii=False, indent=4)

