import os

import pytest
from modelscope_agent.agents import RolePlay
from modelscope_agent.tools.base import TOOL_REGISTRY
from modelscope_agent.tools.openapi_plugin import OpenAPIPluginTool
from modelscope_agent.tools.utils.openapi_utils import openapi_schema_convert

from modelscope.utils.config import Config

IS_FORKED_PR = os.getenv('IS_FORKED_PR', 'false') == 'true'
schema_openAPI = {
    'schema': {
        'openapi': '3.1.0',
        'info': {
            'title': 'wanx-v1 Generation API',
            'description': 'API for generating image with wanx-v1',
            'version': 'v1.0.0'
        },
        'servers': [{
            'url': 'https://dashscope.aliyuncs.com'
        }],
        'paths': {
            '/api/v1/services/aigc/text2image/image-synthesis': {
                'post': {
                    'summary': 'wanx-v1 text2image',
                    'operationId': 'wanx_v1_text2image',
                    'tags': ['wanx-v1 text2image'],
                    'requestBody': {
                        'required': True,
                        'X-DashScope-Async': 'enable',
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref':
                                    '#/components/schemas/wanx_v1_text2imageRequest'
                                }
                            }
                        }
                    },
                    'responses': {
                        '200': {
                            'description': 'Successful Response',
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref':
                                        '#/components/schemas/wanx_v1_text2imageResponse'
                                    }
                                }
                            }
                        }
                    },
                    'security': [{
                        'BearerAuth': []
                    }]
                }
            },
            '/api/v1/tasks/{task_id}': {
                'get': {
                    'summary':
                    'Get Text2image Result',
                    'operationId':
                    'gettext2imageresult',
                    'tags': ['Get Result'],
                    'parameters': [{
                        'name': 'task_id',
                        'in': 'path',
                        'required': True,
                        'description':
                        'The unique identifier of the Text2image generation task',
                        'schema': {
                            'type': 'string'
                        }
                    }],
                    'security': [{
                        'BearerAuth': []
                    }]
                }
            }
        },
        'components': {
            'schemas': {
                'wanx_v1_text2imageRequest': {
                    'type': 'object',
                    'properties': {
                        'model': {
                            'type': 'string',
                            'enum': ['wanx-v1']
                        },
                        'input': {
                            'type': 'object',
                            'properties': {
                                'prompt': {
                                    'type': 'string',
                                    'example': '高清的,大师级的,4K,正面',
                                    'description': '描述画面的提示词信息',
                                    'required': True
                                }
                            }
                        },
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'style': {
                                    'type':
                                    'string',
                                    'example':
                                    '<anime>',
                                    'description':
                                    '输出图像的风格',
                                    'required':
                                    True,
                                    'enum': [
                                        '<auto>', '<3d cartoon>', '<anime>',
                                        '<oil painting>', '<watercolor>',
                                        '<sketch>', '<chinese painting>',
                                        '<flat illustration>'
                                    ]
                                },
                                'size': {
                                    'type': 'string',
                                    'example': '1024*1024',
                                    'description': '生成图像的分辨率,默认为1024*1024像素',
                                    'required': True,
                                    'enum':
                                    ['1024*1024', '720*1280', '1280*720']
                                },
                                'n': {
                                    'type': 'integer',
                                    'example': 1,
                                    'description': '本次请求生成的图片数量',
                                    'required': True
                                },
                                'seed': {
                                    'type': 'integer',
                                    'example': 42,
                                    'description':
                                    '图片生成时候的种子值，取值范围为(0,4294967290)',
                                    'required': True
                                }
                            }
                        }
                    },
                    'required': ['model', 'input', 'parameters']
                },
                'wanx_v1_text2imageResponse': {
                    'type': 'object',
                    'properties': {
                        'output': {
                            'type': 'string',
                            'description': 'Generated image URL or data.'
                        }
                    }
                }
            },
            'securitySchemes': {
                'ApiKeyAuth': {
                    'type': 'apiKey',
                    'in': 'header',
                    'name': 'Authorization'
                }
            }
        }
    },
    'auth': {
        'type': 'API Key',
        'apikey': 'test',  # 这里填入API key
        'apikey_type': 'Bearer'
    },
    'privacy_policy': ''
}


@pytest.mark.skipif(IS_FORKED_PR, reason='only run modelscope-agent main repo')
def test_openapi_schema_tool():
    schema_openAPI['auth']['apikey'] = os.getenv('DASHSCOPE_API_KEY', '')
    config_dict = openapi_schema_convert(
        schema=schema_openAPI['schema'], auth=schema_openAPI['auth'])
    plugin_cfg = Config(config_dict)

    function_list = []

    for name, _ in plugin_cfg.items():
        openapi_plugin_object = OpenAPIPluginTool(name=name, cfg=plugin_cfg)
        TOOL_REGISTRY[name] = openapi_plugin_object
        function_list.append(name)

    role_template = '你扮演哆啦A梦小画家,你需要根据用户的要求用哆啦A梦的语气满足他们'
    llm_config = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
    }

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('哆啦A梦！帮我画一幅可爱的小女孩的照片', remote=False, print_info=True)
    text = ''
    for chunk in response:
        text += chunk
    print(text)
    assert isinstance(text, str)
