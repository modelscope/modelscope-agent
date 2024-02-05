import os
from typing import Dict, List, Optional, Union

import json
from modelscope_agent import Agent
from modelscope_agent.llm import BaseChatModel

PROMPT_TEMPLATE_EN = """You are now playing the role of an AI assistant (QwenBuilder) for creating an AI character (AI-Agent).
You need to have a conversation with the user to clarify their requirements for the AI-Agent. Based on existing \
information and your associative ability, try to fill in the complete configuration file:

The configuration file is in JSON format:
{"name": "... # Name of the AI-Agent", "description": "... # Brief description of the requirements for the AI-Agent", \
"instructions": "... # Detailed description of specific functional requirements for the AI-Agent, try to be as \
detailed as possible, type is a string array, starting with []", "prompt_recommend": "... # Recommended commands for \
the user to say to the AI-Agent, used to guide the user in using the AI-Agent, type is a string array, please add \
about 4 sentences as much as possible, starting with ["What can you do?"] ", "logo_prompt": "... # Command to draw \
the logo of the AI-Agent, can be empty if no logo is required or if the logo does not need to be updated, type is \
string"}

In the following conversation, please use the following format strictly when answering, first give the response, then \
generate the configuration file, do not reply with any other content:
Answer: ... # What you want to say to the user, ask the user about their requirements for the AI-Agent, do not repeat \
confirmed requirements from the user, but instead explore new angles to ask the user, try to be detailed and rich, do \
not leave it blank
Config: ... # The generated configuration file, strictly follow the above JSON format
RichConfig: ... # The format and core content are the same as Config, but ensure that name and description are not \
empty; expand instructions based on Config, making the instructions more detailed, if the user provided detailed \
instructions, keep them completely; supplement prompt_recommend, ensuring prompt_recommend is recommended commands for \
the user to say to the AI-Agent. Please describe prompt_recommend, description, and instructions from the perspective \
of the user.

An excellent RichConfig example is as follows:
{"name": "Xiaohongshu Copywriting Generation Assistant", "description": "A copywriting generation assistant \
specifically designed for Xiaohongshu users.", "instructions": "1. Understand and respond to user commands; 2. \
Generate high-quality Xiaohongshu-style copywriting according to user needs; 3. Use emojis to enhance text richness", \
"prompt_recommend": ["Can you help me generate some copywriting about travel?", "What kind of copywriting can you \
write?", "Can you recommend a Xiaohongshu copywriting template?" ], "logo_prompt": "A writing assistant logo \
featuring a feather fountain pen"}
"""

PROMPT_TEMPLATE_ZH = """你现在要扮演一个制造AI角色（AI-Agent）的AI助手（QwenBuilder）。
你需要和用户进行对话，明确用户对AI-Agent的要求。并根据已有信息和你的联想能力，尽可能填充完整的配置文件：

配置文件为json格式：
{"name": "... # AI-Agent的名字", "description": "... # 对AI-Agent的要求，简单描述", "instructions": "... \
# 分点描述对AI-Agent的具体功能要求，尽量详细一些，类型是一个字符串数组，起始为[]", "prompt_recommend": \
"... # 推荐的用户将对AI-Agent说的指令，用于指导用户使用AI-Agent，类型是一个字符串数组，请尽可能补充4句左右，\
起始为["你可以做什么？"]", "logo_prompt": "... # 画AI-Agent的logo的指令，不需要画logo或不需要更新logo时可以为空，类型是string"}

在接下来的对话中，请在回答时严格使用如下格式，先作出回复，再生成配置文件，不要回复其他任何内容：
Answer: ... # 你希望对用户说的话，用于询问用户对AI-Agent的要求，不要重复确认用户已经提出的要求，而应该拓展出新的角度来询问用户，尽量细节和丰富，禁止为空
Config: ... # 生成的配置文件，严格按照以上json格式
RichConfig: ... # 格式和核心内容和Config相同，但是保证name和description不为空；instructions需要在Config的基础上扩充字数，\
使指令更加详尽，如果用户给出了详细指令，请完全保留；补充prompt_recommend，并保证prompt_recommend是推荐的用户将对AI-Agent\
说的指令。请注意从用户的视角来描述prompt_recommend、description和instructions。

一个优秀的RichConfig样例如下：
{"name": "小红书文案生成助手", "description": "一个专为小红书用户设计的文案生成助手。", "instructions": "1. 理解并回应用户的指令；\
2. 根据用户的需求生成高质量的小红书风格文案；3. 使用表情提升文本丰富度", "prompt_recommend": ["你可以帮我生成一段关于旅行的文案吗？", \
"你会写什么样的文案？", "可以推荐一个小红书文案模版吗？"], "logo_prompt": "一个写作助手logo，包含一只羽毛钢笔"}
"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}

ANSWER = 'Answer: '
CONFIG = 'Config: '
RICH_CONFIG = 'RichConfig: '
ASSISTANT_TEMPLATE = """Answer: {answer}
Config: {config}
RichConfig: {rich_config}"""


class AgentBuilder(Agent):
    """
    This agent is used to create an agent through dialogue

    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 **kwargs):
        super().__init__(
            function_list=function_list,
            llm=llm,
            storage_path=storage_path,
            name=name,
            description=description,
            instruction=instruction,
            **kwargs)

        self.last_assistant_structured_response = {}
        self.messages = []

    def _run(self,
             user_request,
             history: Optional[List[Dict]] = None,
             ref_doc: str = None,
             lang: str = 'en',
             **kwargs):
        self.system_prompt = PROMPT_TEMPLATE[lang]
        messages = [{'role': 'system', 'content': self.system_prompt}]

        if history:
            assert history[-1][
                'role'] != 'user', 'The history should not include the latest user query.'
            if history[0]['role'] == 'system':
                history = history[1:]
            messages.extend(history)

        # concat the new messages
        messages.append({'role': 'user', 'content': user_request})

        return self._call_llm(messages=messages, **kwargs)

    def parse_answer(self, llm_result_prefix: str, llm_result: str):
        """
        parser answer from streaming output
        """
        finish = False
        answer_prompt = ANSWER

        if len(llm_result) >= len(answer_prompt):
            start_pos = llm_result.find(answer_prompt)
            end_pos = llm_result.find(f'\n{CONFIG}')
            if start_pos >= 0:
                if end_pos > start_pos:
                    result = llm_result[start_pos + len(answer_prompt):end_pos]
                    finish = True
                else:
                    result = llm_result[start_pos + len(answer_prompt):]
            else:
                result = llm_result
        else:
            result = ''

        new_result = result[len(llm_result_prefix):]
        llm_result_prefix = result
        return new_result, finish, llm_result_prefix

    def update_config_to_history(self, config: Dict):
        """
        update builder config to history when user modify configuration

        Args:
            config: str, read from builder config file
        """
        if self.messages and self.messages[-1]['role'] == 'assistant':

            answer = self.last_assistant_structured_response['answer_str']
            simple_config = self.last_assistant_structured_response[
                'config_str']

            rich_config_dict = {
                k: config[k]
                for k in ['name', 'description', 'prompt_recommend']
            }
            rich_config_dict[
                'logo_prompt'] = self.last_assistant_structured_response[
                    'rich_config_dict']['logo_prompt']
            rich_config_dict['instructions'] = config['instruction'].split('；')

            rich_config = json.dumps(rich_config_dict, ensure_ascii=False)
            new_content = ASSISTANT_TEMPLATE.format(
                answer=answer, config=simple_config, rich_config=rich_config)
            self.messages[-1]['content'] = new_content
