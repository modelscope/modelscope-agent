import re
from ..memory import memory,LTM
from ..memory.long_term_memroy import DATA_PATH
from pathlib import Path
class PromptGenerator:

    def __init__(self,
                 system_template: str = '',
                 instruction_template: str = '',
                 user_template: str = '',
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

    def init_prompt(self, task, tool_list, knowledge_list):
        """
        in this function, the prompt will be initialized.
        """
        prompt = self.sep.join(
            [self.system_template, self.instruction_template])
        prompt += f'这里是参考的历史信息,若没有信息则无需参考。<long_term_memory><knowledge>这里是多轮对话历史。<history>{self.sep}{self.user_template}'

        knowledge_str = self.get_knowledge_str(knowledge_list)
        # long_term_memory
        existing_files = list(Path(DATA_PATH / 'faiss_index').glob("*.pkl"))  # 获取路径下所有的.pkl文件
        if existing_files:  # 如果存在.pkl文件，表示已有旧的向量库
            long_term_memory_str = LTM().search(task)
        else:
            long_term_memory_str = ""
        prompt = prompt.replace('<long_term_memory>', long_term_memory_str)
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
        #print("wwwwwwww",self.assistant_template)
        # store history
        try:
            self.history = memory(self.user_template,self.assistant_template).store(task)
        except Exception as e:
            print(f"存储工作记忆时发生异常: {e}")
        # self.history.append({
        #     'role':
        #     'user',
        #     'content':
        #     self.user_template.replace('<user_input>', task)
        # })
        # self.history.append({
        #     'role': 'assistant',
        #     'content': self.assistant_template
        # })

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
            exec_result = self.exec_template.replace('<exec_result>',
                                                     str(exec_result))
            self.prompt = f'{self.prompt}{self.sep}{exec_result}'
            self.history[-1]['content'] += f'{self.sep}{exec_result}'
        # print("wwwwww",self.history[-1]['content'])
        #print("wwwwwwwww",self.history)
        return self.prompt,self.history

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
            if len(history_str) + len(text) + len(
                    self.prompt) > self.prompt_max_length:
                break
            history_str = f'{self.sep}{text.strip()}{history_str}'

        return history_str
