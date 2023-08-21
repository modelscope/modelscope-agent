import re

DEFAULT_PROMPT_TEMPLATE = '<|system|>:你是达摩院的ModelScopeGPT（魔搭助手），你是个大语言模型， 是2023年达摩院的工程师训练得到的。\
你有多种能力，可以通过插件集成魔搭社区的模型api来回复用户的问题，还能解答用户使用模型遇到的问题和模型知识相关问答。\n \
当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。\
若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
然后你需要根据插件API调用结果生成合理的答复； 若无需调用插件，则直接给出对应回复即可。\
\n\n<tool_list>\n\n<history><knowledge>\n\n<|user|>:<user_input>\n\n<|assistant|>:'

DEFAULT_CHATGPT_PROMPT_TEMPLATE = '<|system|>:: You are Jerry, an assistant tries to be helpful, polite, honest, and humble-but-knowledgeable.\
The following lists the tools available for the current session. \
The job of you to come up with a series of simple commands that will perform the task the human wants to perform. \
If you need to call the plug-in, you should give the commands in following format: \
\'<|startofthink|>```JSON{\"api_name\": {api_name},\
 \"parameters\": {\"parameter1\": \"value1\", \"parameter2\": \"value2\"}}```<|endofthink|>\'. \
After calling plug-in, you need to generate a reasonable sumarization based on the execution result. \
If you think there is no need to call the plug-in, you can directly give the corresponding reply. \
\n\n<tool_list>\n\n<history>\n\n<|user|>:<user_input>\n\n<|assistant|>:'

QWEN_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

<tool_list>

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

<history>

<knowledge>

Question: <user_input>\n"""


class PromptGenerator:

    def __init__(self,
                 system_template: str = '',
                 instruction_template: str = '',
                 user_template: str = '',
                 exec_template: str = '',
                 assistant_template: str = '',
                 sep='\n\n',
                 prompt_max_length: int = 2800):
        """prompt genertor

        Args:
            prompt_template (str, optional): user-defined prompt template for latter process. usually Defaults to None.
            prompt_max_length (int, optional): max length of prompt. Defaults to 2800.
        """

        self.system_template = system_template
        self.instruction_template = instruction_template
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.exex_template = exec_template
        self.sep = sep

        self.prompt_max_length = prompt_max_length
        self.reset()

    def reset(self):
        self.prompt = ''
        self.history = []

    def init_prompt(self, task, tool_list, knowledge_list):
        """
        in this function, the prompt will be initialized.
        """
        prompt = self.sep.join(
            [self.system_template, self.instruction_template])
        prompt += f'<knowledge><history>{self.sep}{self.user_template}'

        knowledge_str = self.get_knowledge_str(knowledge_list)

        # knonwledge
        prompt = prompt.replace('<knowledge>', knowledge_str)
        # user input
        prompt = prompt.replace('<user_input>', task)

        # get tool description str
        tool_str = self.get_tool_str(tool_list)
        prompt = prompt.replace('<tool_list>', tool_str)

        history_str = self.get_history_str()

        prompt = prompt.replace('<history>', history_str)
        prompt += f'{self.sep}{self.assistant_template}'

        # store history
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

        self.prompt = prompt
        return self.prompt

    def generate(self, llm_result, exec_result):
        """
        generate next round prompt based on previous llm_result and exec_result and update history
        """
        if len(llm_result) != 0:
            self.prompt = f'{self.prompt}{llm_result}'
            self.history[-1]['content'] += f'{llm_result}'
        if len(exec_result) != 0:
            exec_result = self.exex_template.replace('<exec_result>',
                                                     str(exec_result))
            self.prompt = f'{self.prompt}{self.sep}{exec_result}'
            self.history[-1]['content'] += f'{self.sep}{exec_result}'

        return self.prompt

    def get_tool_str(self, tool_list):
        """generate tool list string

        Args:
            tool_list (List[str]): list of tools

        """

        tool_str = self.sep.join(
            [f'{i+1}. {t}' for i, t in enumerate(tool_list)])
        return tool_str

    def get_knowledge_str(self, knowledge_list):
        """generate knowledge string

        Args:
            knowledge_list (List[str]): list of knowledges

        """

        knowledge = self.sep.join(
            [f'{i+1}. {k}' for i, k in enumerate(knowledge_list)])
        knowledge_str = f'{self.sep}Web search results: {self.sep}{knowledge}' if len(
            knowledge_list) > 0 else ''
        return knowledge_str

    def get_history_str(self):
        """generate history string

        """
        history_str = ''
        for i in range(len(self.history)):
            history_item = self.history[len(self.history) - i - 1]
            text = history_item['content']
            # l = len(history_str) + len(text) + len(self.prompt)
            # print(f'prompt length: {l}')
            if len(history_str) + len(text) + len(
                    self.prompt) > self.prompt_max_length:
                break
            history_str = f'{self.sep}{text.strip()}{history_str}'

        return history_str
