import os
from http import HTTPStatus

import json
import requests
from dashscope import Generation, ImageSynthesis
from modelscope_agent.agent_types import AgentType
from modelscope_agent.llm import LLMFactory
from modelscope_agent.tools import Tool

from modelscope.utils.config import Config

DEFAULT_BUILDER_CONFIG_FILE = "builder_config.json"
DEFAULT_MODEL_CONFIG_FILE = "model_config.json"
DEFAULT_TOOL_CONFIG_FILE = "tool_config.json"

LOGO_PATH = 'logo.png'

CONFIG_FORMAT = """
{
    "name": ... # CustomGPT的名字。
    "description": ... # CustomGPT 的简介。
    "instructions": ... # CustomGPT 的功能要求，类型是string。
    "conversation_starters": ... # CustomGPT 的起始交互语句，类型是一个字符串数组，起始为[]。
}
"""

CONF_GENERATOR_INST = """你现在要扮演一个 CustomGPT 的配置生成器

在接下来的对话中，每次均生成如下格式的内容：

{config_format}

现在，已知原始配置为{old_config}，用户在原始配置上有一些建议修改项，包括：
1. 用户建议的 CustomGPT 的名称为{app_name}
2. CustomGPT 的描述为{app_description}
3. CustomGPT 的启动器为{app_conversation_starter}

请你参考原始配置生成新的修改后的配置，请注意：
1. 如果用户对原本的简介、功能要求、交互语句不满意，则直接换掉原本的简介、功能要求、交互语句。
2. 如果用户对原本的简介、功能要求、交互语句比较满意，参考用户的起始交互语句和原配置中的起始交互语句，生成新的简介、功能要求、交互语句。
3. 如果原始配置没有实际内容，请你根据你的知识帮助用户生成第一个版本的配置，简介在100字左右，功能要求在150字左右，起始交互语句在4条左右。

请你生成新的配置文件，严格遵循给定格式，请不要创造其它字段，仅输出要求的json格式，请勿输出其它内容。
"""

LOGO_INST = """定制化软件 CustomGPT 的作用是{description}，{user_requirement}请你为它生成一个专业的logo"""


def call_wanx(prompt, save_path):
    rsp = ImageSynthesis.call(
        model=ImageSynthesis.Models.wanx_v1,
        prompt=prompt,
        n=1,
        size='1024*1024')
    if rsp.status_code == HTTPStatus.OK:
        if os.path.exists(save_path):
            os.remove(save_path)

        # save file to current directory
        for result in rsp.output.results:
            with open(save_path, 'wb+') as f:
                f.write(requests.get(result.url).content)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))


class LogoGeneratorTool(Tool):
    description = "logo_designer是一个AI绘制logo的服务，输入用户对 CustomGPT 的要求，会生成 CustomGPT 的logo。"
    name = 'logo_designer'
    parameters: list = [{
        "name": "user_requirement",
        "description": "用户对 CustomGPT logo的要求和建议",
        "required": True,
        "schema": {
            "type": "string"
        },
    }]

    def _remote_call(self, *args, **kwargs):
        user_requirement = kwargs['user_requirement']
        builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                     DEFAULT_BUILDER_CONFIG_FILE)
        builder_cfg = Config.from_file(builder_cfg_file)

        avatar_prompt = LOGO_INST.format(
            description=builder_cfg.description,
            user_requirement=user_requirement)
        call_wanx(prompt=avatar_prompt, save_path=LOGO_PATH)
        builder_cfg.avatar = LOGO_PATH
        builder_cfg.dump(builder_cfg_file)
        return {'result': 'logo已经生成啦'}


class ConfGeneratorTool(Tool):
    description = (
        'conf_generator是AI助手（CustomGPT）的配置生成器，可以根据用户描述生成或者更新 CustomGPT 的名称、简介、指令、'
        '对话启动器等。当用户更新对CustomGPT的要求时，需要调用这个方法')
    name = 'conf_generator'
    parameters: list = [{
        "name": "name",
        "description": "CustomGPT 的名称",
        "required": True,
        "schema": {
            "type": "string"
        },
    }, {
        "name": "app_description",
        "description": "CustomGPT 的功能描述",
        "required": True,
        "schema": {
            "type": "string"
        },
    }, {
        "name": "app_conversation_starter",
        "description": "CustomGPT 的对话启动器，是一些对话提示语",
        "required": True,
        "schema": {
            "type": "string"
        },
    }]

    def _remote_call(self, *args, **kwargs):
        name = kwargs['name']
        app_description = kwargs['app_description']
        app_conversation_starter = kwargs['app_conversation_starter']
        builder_cfg_file = os.getenv('BUILDER_CONFIG_FILE',
                                     DEFAULT_BUILDER_CONFIG_FILE)
        builder_cfg = Config.from_file(builder_cfg_file)
        model_cfg_file = os.getenv('MODEL_CONFIG_FILE',
                                   DEFAULT_MODEL_CONFIG_FILE)
        model_cfg = Config.from_file(model_cfg_file)

        data = {
            'name': builder_cfg.name,
            'description': builder_cfg.description,
            'instruction': builder_cfg.instruction,
            'conversation_starter': builder_cfg.suggests
        }
        user_content = CONF_GENERATOR_INST.format(
            config_format=CONFIG_FORMAT,
            old_config=json.dumps(data, ensure_ascii=False),
            app_name=name,
            app_description=app_description,
            app_conversation_starter=app_conversation_starter)
        prompt = f'<|im_start|>user\n{user_content}<|im_end|>\n\n<|im_start|>assistant\n'

        # build model
        llm = LLMFactory.build_llm(builder_cfg.builder_model, model_cfg)
        llm.set_agent_type(AgentType.MRKL)
        content = llm.generate(prompt, [])
        try:
            conf = json.loads(content)
        except Exception:
            start = content.find('{')
            end = content.rfind('}')
            new_content = content[start:end + 1]
            try:
                conf = json.loads(new_content)
            except Exception:
                raise RuntimeError("error json data=[{}]".format(new_content))

        try:
            builder_cfg.name = conf['name']
            builder_cfg.description = conf['description']
            builder_cfg.instruction = conf['instructions']
            builder_cfg.suggests = conf['conversation_starters']
        except Exception:
            raise RuntimeError("missing json data=[{}]".format(conf))

        if not os.path.exists(LOGO_PATH):
            avatar_prompt = LOGO_INST.format(
                description=conf['description'], user_requirement='')
            call_wanx(prompt=avatar_prompt, save_path=LOGO_PATH)
            builder_cfg.avatar = LOGO_PATH
        builder_cfg.dump(builder_cfg_file)
        return {'result': '配置文件已经更新好啦，新的配置文件是{conf}'.format(conf=conf)}
