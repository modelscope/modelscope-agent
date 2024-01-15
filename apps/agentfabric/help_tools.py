import os
from http import HTTPStatus

import json
import requests
from config_utils import DEFAULT_BUILDER_CONFIG_DIR, get_user_cfg_file
from dashscope import ImageSynthesis
from modelscope_agent.utils.logger import agent_logger as logger

from modelscope.utils.config import Config

LOGO_NAME = 'custom_bot_avatar.png'
LOGO_PATH = os.path.join(DEFAULT_BUILDER_CONFIG_DIR, LOGO_NAME)

CONFIG_FORMAT = """
{
    "name": ... # CustomGPT的名字。
    "description": ... # CustomGPT 的简介。
    "instructions": ... # CustomGPT 的功能要求，类型是string。
    "prompt_recommend": ... # CustomGPT 的起始交互语句，类型是一个字符串数组，起始为[]。
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


def get_logo_path(uuid_str=''):
    logo_path = os.getenv('LOGO_PATH', LOGO_PATH)
    # convert from ./config/builder_config.json to ./config/user/builder_config.json
    logo_path = logo_path.replace('config/', 'config/user/')

    # convert from ./config/user to ./config/uuid
    if uuid_str != '':
        logo_path = logo_path.replace('user', uuid_str)
    if not os.path.exists(logo_path):
        os.makedirs(os.path.dirname(logo_path), exist_ok=True)
    return logo_path


def call_wanx(prompt, save_path, uuid_str):
    rsp = ImageSynthesis.call(
        model='wanx-lite', prompt=prompt, n=1, size='768*768')
    if rsp.status_code == HTTPStatus.OK:
        if os.path.exists(save_path):
            os.remove(save_path)

        # save file to current directory
        for result in rsp.output.results:
            with open(save_path, 'wb+') as f:
                f.write(requests.get(result.url).content)
    else:
        logger.query_error(
            uuid=uuid_str,
            error='wanx error',
            content={
                'wanx_status_code': rsp.status_code,
                'wanx_code': rsp.code,
                'wanx_message': rsp.message
            })


def logo_generate_remote_call(*args, **kwargs):
    user_requirement = kwargs['user_requirement']
    uuid_str = kwargs.get('uuid_str', '')
    builder_cfg_file = get_user_cfg_file(uuid_str)
    builder_cfg = Config.from_file(builder_cfg_file)

    avatar_prompt = LOGO_INST.format(
        description=builder_cfg.description, user_requirement=user_requirement)
    call_wanx(
        prompt=avatar_prompt,
        save_path=get_logo_path(uuid_str=uuid_str),
        uuid_str=uuid_str)
    builder_cfg.avatar = LOGO_NAME
    return {'result': builder_cfg}


class LogoGeneratorTool:
    description = 'logo_designer是一个AI绘制logo的服务，输入用户对 CustomGPT 的要求，会生成 CustomGPT 的logo。'
    name = 'logo_designer'
    parameters: list = [{
        'name': 'user_requirement',
        'description': '用户对 CustomGPT logo的要求和建议',
        'required': True,
        'schema': {
            'type': 'string'
        },
    }]


def config_conversion(generated_config: dict, save=False, uuid_str=''):
    """
    convert
    {
        name: "铁人",
        description: "我希望我的AI-Agent是一个专业的健身教练，专注于力量训练方面，可以提供相关的建议和指南。
        它还可以帮我跟踪和记录每次的力量训练数据，以及提供相应的反馈和建议，帮助我不断改进和优化我的训练计划。
        此外，我希望它可以拥有一些特殊技能和功能，让它更加实用和有趣。例如，它可以帮助我预测未来的身体状况、分析我的营养摄入情况、
        提供心理支持等等。我相信，在它的帮助下，我可以更快地达到自己的目标，变得更加强壮和健康。",
        instructions: [
            "提供力量训练相关的建议和指南",
            "跟踪和记录每次的力量训练数据",
            "提供反馈和建议，帮助改进和优化训练计划",
            "预测未来的身体状况",
            "分析营养摄入情况",
            "提供心理支持",
        ],
        prompt_recommend: [
            "你好，今天的锻炼计划是什么呢？",
            "你觉得哪种器械最适合练背部肌肉呢？",
            "你觉得我现在的训练强度合适吗？",
            "你觉得哪种食物最适合增肌呢？",
        ],
        logo_prompt: "设计一个肌肉男形象的Logo",
    }
    to
    {
        name: "铁人",
        description: "我希望我的AI-Agent是一个专业的健身教练，专注于力量训练方面，可以提供相关的建议和指南。
        它还可以帮我跟踪和记录每次的力量训练数据，以及提供相应的反馈和建议，帮助我不断改进和优化我的训练计划。
        此外，我希望它可以拥有一些特殊技能和功能，让它更加实用和有趣。例如，它可以帮助我预测未来的身体状况、
        分析我的营养摄入情况、提供心理支持等等。我相信，在它的帮助下，我可以更快地达到自己的目标，变得更加强壮和健康。",
        instructions: "提供力量训练相关的建议和指南；跟踪和记录每次的力量训练数据；提供反馈和建议，帮助改进和优化训练计划；
        预测未来的身体状况；分析营养摄入情况；提供心理支持",
        prompt_recommend: [
            "你好，今天的锻炼计划是什么呢？",
            "你觉得哪种器械最适合练背部肌肉呢？",
            "你觉得我现在的训练强度合适吗？",
            "你觉得哪种食物最适合增肌呢？",
        ],
        tools: xxx
        model: yyy
    }
    :param generated_config:
    :return:
    """
    builder_cfg_file = get_user_cfg_file(uuid_str)
    builder_cfg = Config.from_file(builder_cfg_file)
    try:
        builder_cfg.name = generated_config['name']
        builder_cfg.description = generated_config['description']
        builder_cfg.prompt_recommend = generated_config['prompt_recommend']
        if isinstance(generated_config['instructions'], list):
            builder_cfg.instruction = '；'.join(
                generated_config['instructions'])
        else:
            builder_cfg.instruction = generated_config['instructions']
        if save:
            json.dump(
                builder_cfg.to_dict(),
                open(builder_cfg_file, 'w'),
                indent=2,
                ensure_ascii=False)
        return builder_cfg
    except ValueError as e:
        raise ValueError(f'failed to save the configuration with info: {e}')
