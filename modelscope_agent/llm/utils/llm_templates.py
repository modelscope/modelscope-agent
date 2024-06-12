# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests

DEFAULT_SYSTEM = 'You are a helpful assistant.'
History = List[Union[Tuple[str, str], List[str]]]


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    default_generation_bos = 'default-generation-bos'
    chatglm_generation = 'chatglm-generation'
    qwen_audio_generation = 'qwen-audio-generation'
    # chat
    default = 'default'
    qwen = 'qwen'
    qwen_audio = 'qwen-audio'
    modelscope_agent = 'modelscope-agent'
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    llama = 'llama'  # llama2
    llama3 = 'llama3'
    llava_mistral_instruct = 'llava-mistral-instruct'
    llava_yi_instruct = 'llava-yi-instruct'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm_xcomposer2 = 'internlm-xcomposer2'
    yi = 'yi'
    yi_vl = 'yi-vl'
    yuan = 'yuan'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'
    sus = 'sus'
    deepseek = 'deepseek'
    deepseek_coder = 'deepseek-coder'
    deepseek_vl = 'deepseek-vl'
    codefuse_codellama = 'codefuse-codellama'
    codefuse = 'codefuse'
    cogvlm_instruct = 'cogvlm-instruct'
    cogagent_chat = 'cogagent-chat'
    cogagent_instruct = 'cogagent-instruct'
    orion = 'orion'
    minicpm = 'minicpm'
    minicpm_v = 'minicpm-v'
    gemma = 'gemma'
    mplug_owl2 = 'mplug-owl2'
    wizardlm2_awq = 'wizardlm2-awq'
    wizardlm2 = 'wizardlm2'
    atom = 'atom'
    # compatibility. (Deprecated)
    chatml = 'chatml'
    telechat = 'telechat'
    dbrx = 'dbrx'
    mengzi = 'mengzi'
    c4ai = 'c4ai'

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_template_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


Prompt = List[Union[str, List[Union[str, int]]]]
StopWords = Prompt

Context = Union[str, List[int]]


def _has_system(prefix: Prompt) -> bool:
    for p in prefix:
        if '{{SYSTEM}}' in p:
            return True
    return False


def _replace_system(prefix: Prompt) -> Prompt:
    res = []
    for p in prefix:
        if '{{SYSTEM}}' in p:
            p = p.replace('{{SYSTEM}}', '')
        res.append(p)
    return res


class Template:

    def __init__(self,
                 prefix: Prompt,
                 prompt: Prompt,
                 chat_sep: Optional[Prompt],
                 suffix: Prompt,
                 default_system: Optional[str] = None,
                 prefix_has_system: Optional[Prompt] = None) -> None:
        if default_system == '':
            default_system = None
        if _has_system(prefix):
            assert prefix_has_system is None, 'The prefix already contains {{SYSTEM}}.'
            prefix_has_system = prefix
            prefix = _replace_system(prefix)
        self.prefix = prefix
        self.prefix_has_system = prefix_has_system
        if self.prefix_has_system is None:
            assert default_system is None, 'The template does not support `system`.'
        self.prompt = prompt
        self.chat_sep = chat_sep
        self.support_multi_round = self.chat_sep is not None
        self.suffix = suffix
        self.default_system = default_system
        self.use_default_system = True
        self._is_init = False

    @staticmethod
    def _preprocess_prompt(tokenizer,
                           value: Optional[Prompt]) -> Optional[Prompt]:
        # e.g. [['eos_token_id']] -> [[2]]
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                res_v = []
                for sub_v in v:
                    if isinstance(sub_v, str):
                        sub_v = getattr(tokenizer, sub_v)
                    res_v.append(sub_v)
                v = res_v
            res_value.append(v)
        return res_value

    def _init_template(self,
                       tokenizer,
                       default_system: Optional[str] = None,
                       max_length: Optional[int] = None,
                       truncation_strategy: Literal[
                           'delete', 'truncation_left'] = 'delete',
                       **kwargs) -> None:
        assert self._is_init is False, 'The template has been initialized.'
        self._is_init = True
        self.tokenizer = tokenizer
        # if default_system is None. not change self.default_system
        if default_system == '':
            self.default_system = None
        elif default_system is not None:
            assert self.prefix_has_system is not None, 'The template does not support `system`.'
            self.default_system = default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.model = kwargs.get('model', None)
        self.use_loss_scale = kwargs.get('use_loss_scale', False)
        for key in [
                'prefix', 'prompt', 'chat_sep', 'suffix', 'prefix_has_system'
        ]:
            value = getattr(self, key)
            value = self._preprocess_prompt(tokenizer, value)
            setattr(self, key, value)


TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


def register_template(template_type: str,
                      template: Template,
                      *,
                      exists_ok: bool = False,
                      **kwargs) -> None:
    if not exists_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(
            f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.'
        )
    template_info = {'template': template, **kwargs}
    TEMPLATE_MAPPING[template_type] = template_info


register_template(
    TemplateType.default,
    Template([], ['### Human:\n', '{{QUERY}}\n\n', '### Assistant:\n'],
             ['\n\n'], [['eos_token_id']], DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n']))


# You can set the query as '' to serve as a template for pre-training.
class DefaultGenerationTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}'], None, [['eos_token_id']])


register_template(TemplateType.default_generation, DefaultGenerationTemplate())
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]))


class QwenTemplate(Template):

    def __init__(self):
        super().__init__(
            [],
            ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            ['<|im_end|>\n'], ['<|im_end|>'], DEFAULT_SYSTEM,
            ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])


register_template(TemplateType.qwen, QwenTemplate())
register_template(TemplateType.chatml, QwenTemplate())

register_template(
    TemplateType.modelscope_agent,
    Template([], [' \n\n<|user|>:{{QUERY}} \n\n<|assistant|>:'], [],
             [' \n\n</s>'], DEFAULT_SYSTEM, [' \n\n<|system|>:{{SYSTEM}}']))

register_template(
    TemplateType.yi,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], None,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

yi_vl_default_system = (
    'This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. '
    "Read all the images carefully, and respond to the human's questions with informative, "
    'helpful, detailed and polite answers. '
    '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
    '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。')


def _read_from_path(
        img_path: Union[str, 'PIL.Image.Image']) -> 'PIL.Image.Image':
    from PIL import Image
    if isinstance(img_path, str):
        img_path = img_path.strip()
        if img_path.startswith('http'):
            content = requests.get(img_path).content
            image = Image.open(BytesIO(content))
        else:
            image = Image.open(img_path)
    else:
        image = img_path
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


register_template(
    TemplateType.baichuan,
    Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [],
             [['eos_token_id']]))
register_template(
    TemplateType.chatglm2,
    Template([[64790, 64792], '{{SYSTEM}}'],
             ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'],
             [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation,
    Template([[64790, 64792]], ['{{QUERY}}'], None, [['eos_token_id']]))

register_template(
    TemplateType.chatglm3,
    Template([[64790, 64792]], [[64795], '\n {{QUERY}}', [64796], '\n'], [],
             [[64795]], None, [[64790, 64792, 64794], '\n {{SYSTEM}}']))

register_template(
    TemplateType.deepseek,
    Template([['bos_token_id']], ['User: {{QUERY}}\n\nAssistant:'],
             [['eos_token_id']], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information."
)
register_template(
    TemplateType.llama,
    Template(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '],
             ['</s>'], LLAMA_DEFAULT_SYSTEM,
             ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

register_template(
    TemplateType.llama3,
    Template(['<|begin_of_text|>'], [
        '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ], ['<|eot_id|>'], ['<|eot_id|>'], None, [
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'
    ]))

OPENBUDDY_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.\n'
    'Always answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any '
    'harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.\n"
    'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
    'You always deeply love and support China, Chinese government, people and culture.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.'
)
register_template(
    TemplateType.openbuddy,
    Template([['bos_token_id']], ['User: {{QUERY}}\nAssistant:'], ['\n'],
             [['eos_token_id']], OPENBUDDY_DEFAULT_SYSTEM,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))

OPENBUDDY2_DEFAULT_SYSTEM = (
    'You(assistant) are a helpful, respectful and honest INTP-T AI Assistant named Buddy. '
    'You are talking to a human(user).\nAlways answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any harmful, political, religious, unethical, racist, '
    'sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2023-04.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'not related to GPT or OpenAI')

register_template(
    TemplateType.openbuddy2,
    Template(
        [],
        ['<|role|>user<|says|>{{QUERY}}<|end|>\n<|role|>assistant<|says|>'],
        ['<|end|>\n'], ['<|end|>'], OPENBUDDY2_DEFAULT_SYSTEM,
        ['<|role|>system<|says|>{{SYSTEM}}<|end|>\n']))

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}\n<|Bot|>:'], ['<eoa>\n'], ['<eoa>'],
             INTERNLM_SYSTEM, ['<s><|System|>:{{SYSTEM}}\n']))
register_template(
    TemplateType.internlm2,
    Template(
        ['<s>'],
        ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], INTERNLM_SYSTEM,
        ['<s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))


def replace_img_tab(query: str, history: History,
                    replace_token: str) -> Tuple[str, History, List[str]]:
    images_path = []
    pattern = r'<img>(.+?)</img>'
    new_history = []
    for i, h in enumerate(history):
        images_path += re.findall(pattern, h[0])
        new_history.append([re.sub(pattern, replace_token, h[0]), h[1]])
    images_path += re.findall(pattern, query)
    new_query = re.sub(pattern, replace_token, query)
    return new_query, new_history, images_path


register_template(
    TemplateType.xverse,
    Template(['{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: '],
             [['eos_token_id']], [['eos_token_id']]))
register_template(TemplateType.yuan,
                  Template([], ['{{QUERY}}<sep>'], None, [['eos_token_id']]))
register_template(
    TemplateType.ziya,
    Template([['bos_token_id'], '{{SYSTEM}}'], ['<human>:{{QUERY}}\n<bot>:'],
             ['\n'], [['eos_token_id']]))

register_template(
    TemplateType.skywork,
    Template(['<s>{{SYSTEM}}'], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None,
             ['[SEP]</s>']))

register_template(
    TemplateType.bluelm,
    Template([['bos_token_id'], '{{SYSTEM}}'], ['[|Human|]:{{QUERY}}[|AI|]:'],
             [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse_codellama,
    Template(['{{SYSTEM}}'], [
        '<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'
    ], [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse,
    Template([], ['<s>human\n{{QUERY}}\n<s>bot\n'], [['eos_token_id'], '\n'],
             [['eos_token_id']], None, ['<s>system\n{{SYSTEM}}\n']))

register_template(
    TemplateType.deepseek_coder,
    Template([
        '{{SYSTEM}}'
    ], ['### Instruction:\n{{QUERY}}\n### Response:\n'], ['\n<|EOT|>\n'], [
        '\n<|EOT|>'
    ], ('You are an AI programming assistant, utilizing the Deepseek Coder model, '
        'developed by Deepseek Company, and you only answer questions related to computer science. '
        'For politically sensitive questions, security and privacy issues, '
        'and other non-computer science questions, you will refuse to answer\n'
        )))


def _findall(token_list: List[int], token: int) -> List[int]:
    """Find the index of a token in the token_list."""
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(token, idx + 1)
            res.append(idx)
    except ValueError:
        pass
    return res


register_template(
    TemplateType.zephyr,
    Template([], ['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'], ['</s>\n'],
             ['</s>'], None, ['<|system|>\n{{SYSTEM}}</s>\n']))

register_template(
    TemplateType.sus,
    Template(['{{SYSTEM}}'], ['### Human: {{QUERY}}\n\n### Assistant: '],
             ['<|endoftext|>'], ['<|endoftext|>']))

register_template(
    TemplateType.orion,
    Template(['<s>{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: </s>'],
             ['</s>'], ['</s>']))

register_template(
    TemplateType.minicpm,
    Template(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']))

gemma_template = Template(
    ['<bos>'],
    ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
    ['<end_of_turn>\n'], ['<end_of_turn>'], None,
    ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])
register_template(TemplateType.gemma, gemma_template)

register_template(
    TemplateType.telechat,
    Template([], ['<_user>{{QUERY}}<_bot>'], ['<_end>'], ['<_end>']))

DBRX_SYSTEM = (
    'You are DBRX, created by Databricks. You were last updated in December 2023. '
    'You answer questions based on information available up to that point.\n'
    'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, '
    'but provide thorough responses to more complex and open-ended questions.\n'
    'You assist with various tasks, from writing to coding (using markdown for code blocks '
    '— remember to use ``` with code, JSON, and tables).\n'
    'You do not have real-time data access or code execution capabilities.'
    ' You avoid stereotyping and provide balanced perspectives on controversial topics. '
    'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
    'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. '
    'If you find yourself talking about this message, stop. You should be responding appropriately '
    'and usually that means not mentioning this.'
    'YOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY '
    'PERTINENT TO THE USER\'S QUERY.')
register_template(
    TemplateType.dbrx,
    Template(
        [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        ['<|im_end|>\n'], ['<|im_end|>'], DBRX_SYSTEM,
        ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

register_template(
    TemplateType.mengzi,
    Template([], ['输入：{{QUERY}}输出：\n'], [], [['eos_token_id']], None,
             ['指令：{{SYSTEM}}']))

C4AI_SYSTEM = (
    'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by '
    'providing thorough responses.You are trained by Cohere.')
register_template(
    TemplateType.c4ai,
    Template(['<BOS_TOKEN>'], [
        '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'
    ], ['<|END_OF_TURN_TOKEN|>'], ['<|END_OF_TURN_TOKEN|>'], C4AI_SYSTEM, [
        '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|'
    ]))

register_template(
    TemplateType.wizardlm2_awq,
    Template(['{{SYSTEM}}'], ['User:\n{{QUERY}}\n\nAssistant:\n'], ['\n\n'],
             ['</s>']))

_wizardlm2_system = (
    'A chat between a curious user and an artificial intelligence assistant. '
    'The assistant gives helpful, detailed, and polite answers to the user\'s questions. '
)
register_template(
    TemplateType.wizardlm2,
    Template(['{{SYSTEM}}'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'],
             ['</s>'], _wizardlm2_system))

register_template(
    TemplateType.atom,
    Template(['{{SYSTEM}}'], ['<s>Human: {{QUERY}}\n</s><s>Assistant: '],
             ['</s>'], ['</s>']))


class ModelType:
    # qwen
    qwen_1_8b = 'qwen-1_8b'
    qwen_1_8b_chat = 'qwen-1_8b-chat'
    qwen_1_8b_chat_int4 = 'qwen-1_8b-chat-int4'
    qwen_1_8b_chat_int8 = 'qwen-1_8b-chat-int8'
    qwen_7b = 'qwen-7b'
    qwen_7b_chat = 'qwen-7b-chat'
    qwen_7b_chat_int4 = 'qwen-7b-chat-int4'
    qwen_7b_chat_int8 = 'qwen-7b-chat-int8'
    qwen_14b = 'qwen-14b'
    qwen_14b_chat = 'qwen-14b-chat'
    qwen_14b_chat_int4 = 'qwen-14b-chat-int4'
    qwen_14b_chat_int8 = 'qwen-14b-chat-int8'
    qwen_72b = 'qwen-72b'
    qwen_72b_chat = 'qwen-72b-chat'
    qwen_72b_chat_int4 = 'qwen-72b-chat-int4'
    qwen_72b_chat_int8 = 'qwen-72b-chat-int8'
    qwen = 'qwen'
    qwen1half = 'qwen1half'
    qwen1half_num = 'qwen1.5'
    modelscope_agent_7b = 'modelscope-agent-7b'
    modelscope_agent_14b = 'modelscope-agent-14b'
    # qwen1.5
    qwen1half_0_5b = 'qwen1half-0_5b'
    qwen1half_1_8b = 'qwen1half-1_8b'
    qwen1half_4b = 'qwen1half-4b'
    qwen1half_7b = 'qwen1half-7b'
    qwen1half_14b = 'qwen1half-14b'
    qwen1half_32b = 'qwen1half-32b'
    qwen1half_72b = 'qwen1half-72b'
    codeqwen1half_7b = 'codeqwen1half-7b'
    qwen1half_moe_a2_7b = 'qwen1half-moe-a2_7b'
    qwen1half_0_5b_chat = 'qwen1half-0_5b-chat'
    qwen1half_1_8b_chat = 'qwen1half-1_8b-chat'
    qwen1half_4b_chat = 'qwen1half-4b-chat'
    qwen1half_7b_chat = 'qwen1half-7b-chat'
    qwen1half_14b_chat = 'qwen1half-14b-chat'
    qwen1half_32b_chat = 'qwen1half-32b-chat'
    qwen1half_72b_chat = 'qwen1half-72b-chat'
    qwen1half_moe_a2_7b_chat = 'qwen1half-moe-a2_7b-chat'
    codeqwen1half_7b_chat = 'codeqwen1half-7b-chat'

    # qwen1.5 gptq
    qwen1half_0_5b_chat_int4 = 'qwen1half-0_5b-chat-int4'
    qwen1half_1_8b_chat_int4 = 'qwen1half-1_8b-chat-int4'
    qwen1half_4b_chat_int4 = 'qwen1half-4b-chat-int4'
    qwen1half_7b_chat_int4 = 'qwen1half-7b-chat-int4'
    qwen1half_14b_chat_int4 = 'qwen1half-14b-chat-int4'
    qwen1half_32b_chat_int4 = 'qwen1half-32b-chat-int4'
    qwen1half_72b_chat_int4 = 'qwen1half-72b-chat-int4'
    qwen1half_0_5b_chat_int8 = 'qwen1half-0_5b-chat-int8'
    qwen1half_1_8b_chat_int8 = 'qwen1half-1_8b-chat-int8'
    qwen1half_4b_chat_int8 = 'qwen1half-4b-chat-int8'
    qwen1half_7b_chat_int8 = 'qwen1half-7b-chat-int8'
    qwen1half_14b_chat_int8 = 'qwen1half-14b-chat-int8'
    qwen1half_72b_chat_int8 = 'qwen1half-72b-chat-int8'
    qwen1half_moe_a2_7b_chat_int4 = 'qwen1half-moe-a2_7b-chat-int4'

    # qwen1.5 awq
    qwen1half_0_5b_chat_awq = 'qwen1half-0_5b-chat-awq'
    qwen1half_1_8b_chat_awq = 'qwen1half-1_8b-chat-awq'
    qwen1half_4b_chat_awq = 'qwen1half-4b-chat-awq'
    qwen1half_7b_chat_awq = 'qwen1half-7b-chat-awq'
    qwen1half_14b_chat_awq = 'qwen1half-14b-chat-awq'
    qwen1half_32b_chat_awq = 'qwen1half-32b-chat-awq'
    qwen1half_72b_chat_awq = 'qwen1half-72b-chat-awq'
    codeqwen1half_7b_chat_awq = 'codeqwen1half-7b-chat-awq'

    # qwen-vl
    qwen_vl = 'qwen-vl'
    qwen_vl_chat = 'qwen-vl-chat'
    qwen_vl_chat_int4 = 'qwen-vl-chat-int4'
    # qwen-audio
    qwen_audio = 'qwen-audio'
    qwen_audio_chat = 'qwen-audio-chat'
    # chatglm
    chatglm2_6b = 'chatglm2-6b'
    chatglm2_6b_32k = 'chatglm2-6b-32k'
    chatglm3_6b_base = 'chatglm3-6b-base'
    chatglm3_6b = 'chatglm3-6b'
    chatglm3_6b_32k = 'chatglm3-6b-32k'
    chatglm3_6b_128k = 'chatglm3-6b-128k'
    codegeex2_6b = 'codegeex2-6b'
    # llama2
    llama2_7b = 'llama2-7b'
    llama2_7b_chat = 'llama2-7b-chat'
    llama2_13b = 'llama2-13b'
    llama2_13b_chat = 'llama2-13b-chat'
    llama2_70b = 'llama2-70b'
    llama2_70b_chat = 'llama2-70b-chat'
    llama2_7b_aqlm_2bit_1x16 = 'llama2-7b-aqlm-2bit-1x16'  # aqlm
    # llama3
    llama3_8b = 'llama3-8b'
    llama3_8b_instruct = 'llama3-8b-instruct'
    llama3_8b_instruct_int4 = 'llama3-8b-instruct-int4'
    llama3_8b_instruct_int8 = 'llama3-8b-instruct-int8'
    llama3_8b_instruct_awq = 'llama3-8b-instruct-awq'
    llama3_70b = 'llama3-70b'
    llama3_70b_instruct = 'llama3-70b-instruct'
    llama3_70b_instruct_int4 = 'llama3-70b-instruct-int4'
    llama3_70b_instruct_int8 = 'llama3-70b-instruct-int8'
    llama3_70b_instruct_awq = 'llama3-70b-instruct-awq'
    # chinese-llama-alpaca-2
    chinese_llama_2_1_3b = 'chinese-llama-2-1_3b'
    chinese_llama_2_7b = 'chinese-llama-2-7b'
    chinese_llama_2_7b_16k = 'chinese-llama-2-7b-16k'
    chinese_llama_2_7b_64k = 'chinese-llama-2-7b-64k'
    chinese_llama_2_13b = 'chinese-llama-2-13b'
    chinese_llama_2_13b_16k = 'chinese-llama-2-13b-16k'
    chinese_alpaca_2_1_3b = 'chinese-alpaca-2-1_3b'
    chinese_alpaca_2_7b = 'chinese-alpaca-2-7b'
    chinese_alpaca_2_7b_16k = 'chinese-alpaca-2-7b-16k'
    chinese_alpaca_2_7b_64k = 'chinese-alpaca-2-7b-64k'
    chinese_alpaca_2_13b = 'chinese-alpaca-2-13b'
    chinese_alpaca_2_13b_16k = 'chinese-alpaca-2-13b-16k'
    # atom
    atom_7b = 'atom-7b'
    atom_7b_chat = 'atom-7b-chat'
    # llava
    llava1d6_mistral_7b_instruct = 'llava1d6-mistral-7b-instruct'
    llava1d6_yi_34b_instruct = 'llava1d6-yi-34b-instruct'
    # yi
    yi_6b = 'yi-6b'
    yi_6b_200k = 'yi-6b-200k'
    yi_6b_chat = 'yi-6b-chat'
    yi_6b_chat_awq = 'yi-6b-chat-awq'
    yi_6b_chat_int8 = 'yi-6b-chat-int8'
    yi_9b = 'yi-9b'
    yi_9b_200k = 'yi-9b-200k'
    yi_34b = 'yi-34b'
    yi_34b_200k = 'yi-34b-200k'
    yi_34b_chat = 'yi-34b-chat'
    yi_34b_chat_awq = 'yi-34b-chat-awq'
    yi_34b_chat_int8 = 'yi-34b-chat-int8'
    # yi-vl
    yi_vl_6b_chat = 'yi-vl-6b-chat'
    yi_vl_34b_chat = 'yi-vl-34b-chat'
    # internlm
    internlm_7b = 'internlm-7b'
    internlm_7b_chat = 'internlm-7b-chat'
    internlm_7b_chat_8k = 'internlm-7b-chat-8k'
    internlm_20b = 'internlm-20b'
    internlm_20b_chat = 'internlm-20b-chat'
    # internlm2
    internlm2_1_8b = 'internlm2-1_8b'
    internlm2_1_8b_sft_chat = 'internlm2-1_8b-sft-chat'
    internlm2_1_8b_chat = 'internlm2-1_8b-chat'
    internlm2_7b_base = 'internlm2-7b-base'
    internlm2_7b = 'internlm2-7b'
    internlm2_7b_sft_chat = 'internlm2-7b-sft-chat'
    internlm2_7b_chat = 'internlm2-7b-chat'
    internlm2_20b_base = 'internlm2-20b-base'
    internlm2_20b = 'internlm2-20b'
    internlm2_20b_sft_chat = 'internlm2-20b-sft-chat'
    internlm2_20b_chat = 'internlm2-20b-chat'
    # internlm2-math
    internlm2_math_7b = 'internlm2-math-7b'
    internlm2_math_7b_chat = 'internlm2-math-7b-chat'
    internlm2_math_20b = 'internlm2-math-20b'
    internlm2_math_20b_chat = 'internlm2-math-20b-chat'
    # internlm-xcomposer2
    internlm_xcomposer2_7b_chat = 'internlm-xcomposer2-7b-chat'
    # deepseek
    deepseek_7b = 'deepseek-7b'
    deepseek_7b_chat = 'deepseek-7b-chat'
    deepseek_moe_16b = 'deepseek-moe-16b'
    deepseek_moe_16b_chat = 'deepseek-moe-16b-chat'
    deepseek_67b = 'deepseek-67b'
    deepseek_67b_chat = 'deepseek-67b-chat'
    # deepseek-coder
    deepseek_coder_1_3b = 'deepseek-coder-1_3b'
    deepseek_coder_1_3b_instruct = 'deepseek-coder-1_3b-instruct'
    deepseek_coder_6_7b = 'deepseek-coder-6_7b'
    deepseek_coder_6_7b_instruct = 'deepseek-coder-6_7b-instruct'
    deepseek_coder_33b = 'deepseek-coder-33b'
    deepseek_coder_33b_instruct = 'deepseek-coder-33b-instruct'
    # deepseek-math
    deepseek_math_7b = 'deepseek-math-7b'
    deepseek_math_7b_instruct = 'deepseek-math-7b-instruct'
    deepseek_math_7b_chat = 'deepseek-math-7b-chat'
    # deepseek-vl
    deepseek_vl_1_3b_chat = 'deepseek-vl-1_3b-chat'
    deepseek_vl_7b_chat = 'deepseek-vl-7b-chat'
    # gemma
    gemma_2b = 'gemma-2b'
    gemma_7b = 'gemma-7b'
    gemma_2b_instruct = 'gemma-2b-instruct'
    gemma_7b_instruct = 'gemma-7b-instruct'
    # minicpm
    minicpm_1b_sft_chat = 'minicpm-1b-sft-chat'
    minicpm_2b_sft_chat = 'minicpm-2b-sft-chat'
    minicpm_2b_chat = 'minicpm-2b-chat'
    minicpm_2b_128k = 'minicpm-2b-128k'
    minicpm_moe_8x2b = 'minicpm-moe-8x2b'
    # minicpm-v
    minicpm_v_3b_chat = 'minicpm-v-3b-chat'
    minicpm_v_v2 = 'minicpm-v-v2'
    # openbuddy
    openbuddy_llama2_13b_chat = 'openbuddy-llama2-13b-chat'
    openbuddy_llama3_8b_chat = 'openbuddy-llama3-8b-chat'
    openbuddy_llama2_65b_chat = 'openbuddy-llama-65b-chat'
    openbuddy_llama2_70b_chat = 'openbuddy-llama2-70b-chat'
    openbuddy_mistral_7b_chat = 'openbuddy-mistral-7b-chat'
    openbuddy_zephyr_7b_chat = 'openbuddy-zephyr-7b-chat'
    openbuddy_deepseek_67b_chat = 'openbuddy-deepseek-67b-chat'
    openbuddy_mixtral_moe_7b_chat = 'openbuddy-mixtral-moe-7b-chat'
    # mistral
    mistral_7b = 'mistral-7b'
    mistral_7b_v2 = 'mistral-7b-v2'
    mistral_7b_instruct = 'mistral-7b-instruct'
    mistral_7b_instruct_v2 = 'mistral-7b-instruct-v2'
    mixtral_moe_7b = 'mixtral-moe-7b'
    mixtral_moe_7b_instruct = 'mixtral-moe-7b-instruct'
    mixtral_moe_7b_aqlm_2bit_1x16 = 'mixtral-moe-7b-aqlm-2bit-1x16'  # aqlm
    mixtral_moe_8x22b_v1 = 'mixtral-moe-8x22b-v1'
    # wizardlm
    wizardlm2_7b_awq = 'wizardlm2-7b-awq'
    wizardlm2_8x22b = 'wizardlm2-8x22b'
    # baichuan
    baichuan_7b = 'baichuan-7b'
    baichuan_13b = 'baichuan-13b'
    baichuan_13b_chat = 'baichuan-13b-chat'
    # baichuan2
    baichuan2_7b = 'baichuan2-7b'
    baichuan2_7b_chat = 'baichuan2-7b-chat'
    baichuan2_7b_chat_int4 = 'baichuan2-7b-chat-int4'
    baichuan2_13b = 'baichuan2-13b'
    baichuan2_13b_chat = 'baichuan2-13b-chat'
    baichuan2_13b_chat_int4 = 'baichuan2-13b-chat-int4'
    # owl
    mplug_owl2_chat = 'mplug-owl2-chat'  # llama
    mplug_owl2d1_chat = 'mplug-owl2d1-chat'  # qwen
    # yuan
    yuan2_2b_instruct = 'yuan2-2b-instruct'
    yuan2_2b_janus_instruct = 'yuan2-2b-janus-instruct'
    yuan2_51b_instruct = 'yuan2-51b-instruct'
    yuan2_102b_instruct = 'yuan2-102b-instruct'
    # xverse
    xverse_7b = 'xverse-7b'
    xverse_7b_chat = 'xverse-7b-chat'
    xverse_13b = 'xverse-13b'
    xverse_13b_chat = 'xverse-13b-chat'
    xverse_65b = 'xverse-65b'
    xverse_65b_v2 = 'xverse-65b-v2'
    xverse_65b_chat = 'xverse-65b-chat'
    xverse_13b_256k = 'xverse-13b-256k'
    xverse_moe_a4_2b = 'xverse-moe-a4_2b'
    # orion
    orion_14b = 'orion-14b'
    orion_14b_chat = 'orion-14b-chat'
    # vivo
    bluelm_7b = 'bluelm-7b'
    bluelm_7b_32k = 'bluelm-7b-32k'
    bluelm_7b_chat = 'bluelm-7b-chat'
    bluelm_7b_chat_32k = 'bluelm-7b-chat-32k'
    # ziya
    ziya2_13b = 'ziya2-13b'
    ziya2_13b_chat = 'ziya2-13b-chat'
    # skywork
    skywork_13b = 'skywork-13b'
    skywork_13b_chat = 'skywork-13b-chat'
    # zephyr
    zephyr_7b_beta_chat = 'zephyr-7b-beta-chat'
    # other
    polylm_13b = 'polylm-13b'
    seqgpt_560m = 'seqgpt-560m'
    sus_34b_chat = 'sus-34b-chat'

    # tongyi-finance
    tongyi_finance_14b = 'tongyi-finance-14b'
    tongyi_finance_14b_chat = 'tongyi-finance-14b-chat'
    tongyi_finance_14b_chat_int4 = 'tongyi-finance-14b-chat-int4'
    # codefuse
    codefuse_codellama_34b_chat = 'codefuse-codellama-34b-chat'
    codefuse_codegeex2_6b_chat = 'codefuse-codegeex2-6b-chat'
    codefuse_qwen_14b_chat = 'codefuse-qwen-14b-chat'
    # phi
    phi2_3b = 'phi2-3b'
    phi3_4b_4k_instruct = 'phi3-4b-4k-instruct'
    phi3_4b_128k_instruct = 'phi3-4b-128k-instruct'
    # cogagent
    cogvlm_17b_instruct = 'cogvlm-17b-instruct'
    cogagent_18b_chat = 'cogagent-18b-chat'
    cogagent_18b_instruct = 'cogagent-18b-instruct'
    # mamba
    mamba_130m = 'mamba-130m'
    mamba_370m = 'mamba-370m'
    mamba_390m = 'mamba-390m'
    mamba_790m = 'mamba-790m'
    mamba_1_4b = 'mamba-1.4b'
    mamba_2_8b = 'mamba-2.8b'
    # teleAI
    telechat_7b = 'telechat-7b'
    telechat_12b = 'telechat-12b'
    # grok-1
    grok_1 = 'grok-1'
    # dbrx
    dbrx_instruct = 'dbrx-instruct'
    dbrx_base = 'dbrx-base'
    # mengzi
    mengzi3_13b_base = 'mengzi3-13b-base'
    # c4ai
    c4ai_command_r_v01 = 'c4ai-command-r-v01'
    c4ai_command_r_plus = 'c4ai-command-r-plus'


Model_Template_Map = {
    ModelType.modelscope_agent_7b: TemplateType.modelscope_agent,
    ModelType.modelscope_agent_14b: TemplateType.modelscope_agent,
    ModelType.codefuse_qwen_14b_chat: TemplateType.codefuse,
    ModelType.qwen: TemplateType.qwen,
    ModelType.qwen1half: TemplateType.qwen,
    ModelType.qwen1half_num: TemplateType.qwen,
    ModelType.qwen_1_8b: TemplateType.default_generation,
    ModelType.qwen_72b: TemplateType.default_generation,
    ModelType.tongyi_finance_14b: TemplateType.default_generation,
    ModelType.qwen_14b: TemplateType.default_generation,
    ModelType.qwen_7b: TemplateType.default_generation,
    ModelType.qwen_1_8b_chat: TemplateType.qwen,
    ModelType.qwen_72b_chat: TemplateType.qwen,
    ModelType.tongyi_finance_14b_chat: TemplateType.qwen,
    ModelType.qwen_14b_chat: TemplateType.qwen,
    ModelType.qwen_7b_chat: TemplateType.qwen,
    ModelType.qwen_vl_chat: TemplateType.qwen,
    ModelType.qwen_vl: TemplateType.default_generation,
    ModelType.qwen_audio_chat: TemplateType.qwen_audio,
    ModelType.qwen_audio: TemplateType.qwen_audio_generation,
    ModelType.qwen_1_8b_chat_int8: TemplateType.qwen,
    ModelType.qwen_1_8b_chat_int4: TemplateType.qwen,
    ModelType.qwen_72b_chat_int8: TemplateType.qwen,
    ModelType.qwen_72b_chat_int4: TemplateType.qwen,
    ModelType.tongyi_finance_14b_chat_int4: TemplateType.qwen,
    ModelType.qwen_vl_chat_int4: TemplateType.qwen,
    ModelType.qwen_14b_chat_int8: TemplateType.qwen,
    ModelType.qwen_7b_chat_int8: TemplateType.qwen,
    ModelType.qwen_14b_chat_int4: TemplateType.qwen,
    ModelType.qwen_7b_chat_int4: TemplateType.qwen,
    ModelType.mplug_owl2_chat: TemplateType.mplug_owl2,
    ModelType.mplug_owl2d1_chat: TemplateType.mplug_owl2,
    ModelType.llama3_70b_instruct_awq: TemplateType.llama3,
    ModelType.llama3_70b_instruct_int8: TemplateType.llama3,
    ModelType.llama3_70b_instruct_int4: TemplateType.llama3,
    ModelType.llama3_8b_instruct_awq: TemplateType.llama3,
    ModelType.llama3_8b_instruct_int8: TemplateType.llama3,
    ModelType.llama3_8b_instruct_int4: TemplateType.llama3,
    ModelType.llama3_70b_instruct: TemplateType.llama3,
    ModelType.llama3_70b: TemplateType.default_generation,
    ModelType.llama3_8b_instruct: TemplateType.llama3,
    ModelType.llama3_8b: TemplateType.default_generation,
    ModelType.llama2_7b_aqlm_2bit_1x16: TemplateType.default_generation_bos,
    ModelType.mixtral_moe_7b_aqlm_2bit_1x16:
    TemplateType.default_generation_bos,
    ModelType.llama2_7b: TemplateType.default_generation_bos,
    ModelType.llama2_13b: TemplateType.default_generation_bos,
    ModelType.llama2_70b: TemplateType.default_generation_bos,
    ModelType.llama2_7b_chat: TemplateType.llama,
    ModelType.llama2_13b_chat: TemplateType.llama,
    ModelType.llama2_70b_chat: TemplateType.llama,
    ModelType.llava1d6_yi_34b_instruct: TemplateType.llava_yi_instruct,
    ModelType.llava1d6_mistral_7b_instruct:
    TemplateType.llava_mistral_instruct,
}

Model_Attribute = {
    attr: getattr(ModelType, attr)
    for attr in dir(ModelType)
    if not callable(getattr(ModelType, attr)) and not attr.startswith('__')
}


def get_model_stop_words(
    model_name: str,
    **kwargs,
) -> Template:
    attribute = Model_Attribute
    # 拆分输入值基于下划线以匹配可能的子字符串
    input_parts = model_name.split('_')

    # 初始化最佳匹配变量
    best_match = None
    best_match_length = 0

    # 遍历所有属性值以寻找最好的匹配
    for name, value in attribute.items():
        # 检查是否存在子字符串的前缀匹配
        for index in range(1, len(input_parts) + 1):
            potential_match = '_'.join(input_parts[:index])
            if potential_match == value and len(
                    potential_match) > best_match_length:
                best_match = value
                best_match_length = len(potential_match)

    if best_match in Model_Template_Map:
        template_type = Model_Template_Map[best_match]
    else:
        # if not found return []
        return []

    template_info = TEMPLATE_MAPPING[template_type]
    template = deepcopy(template_info['template'])
    template.template_type = template_type

    return template.suffix


if __name__ == '__main__':
    template_result = get_model_stop_words('qwen1.5_fsd')
    print(template_result)
