import os
import sys
import time

import json
import yaml
from modelscope_agent.tools.openapi_tool import (OpenAPISchemaTool,
                                                 openapi_schema_convert)

from modelscope.utils.config import Config

sys.path.append('../../')

file_path = '../../apps/agentfabric/config/additional_tool_config.json'


def is_json(data):
    try:
        json.loads(data)
        return True
    except ValueError:
        return False


def is_yaml(data):
    try:
        yaml.safe_load(data)
        return True
    except yaml.YAMLError:
        return False


def test_openapi_schema_convert(token):

    schema = """
{
    "openapi": "3.1.0",
    "info": {
      "title": "WordArt Semantic Generation API",
      "description": "API for generating semantic word art with customizable parameters.",
      "version": "v1.0.0"
    },
    "servers": [
      {
        "url": "https://dashscope.aliyuncs.com"
      }
    ],
    "paths": {
      "/api/v1/services/aigc/wordart/semantic": {
        "post": {
          "summary": "Generate WordArt Semantically",
          "operationId": "generateWordArt",
          "tags": [
            "WordArt Generation"
          ],
          "requestBody": {
            "required": true,
            "X-DashScope-Async": "enable",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/WordArtGenerationRequest"
                }
              }
            }
          },
          "responses": {
            "200": {
              "description": "Successful Response",
              "content": {
                "application/json": {
                  "schema": {
                    "$ref": "#/components/schemas/WordArtGenerationResponse"
                  }
                }
              }
            }
          },
          "security": [
            {
              "BearerAuth": []
            }
          ]
        }
      },
      "/api/v1/tasks/{task_id}": {
        "get": {
          "summary": "Get WordArt Result",
          "operationId": "getwordartresult",
          "tags": [
            "Get Result"
          ],
          "parameters": [],
          "security": [
            {
              "BearerAuth": []
            }
          ]

        }
      }
    },
    "components": {
      "schemas": {
        "WordArtGenerationRequest": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "enum": ["wordart-semantic"]
            },
            "input": {
              "type": "object",
              "properties":{
                "text": {
                    "type": "string",
                    "example": "文字创意"
                  },
                  "prompt": {
                    "type": "string",
                    "example": "水果，蔬菜，温暖的色彩空间"
                  }
              }
            },
            "parameters": {
              "type": "object",
              "properties": {
                "steps": {
                  "type": "integer",
                  "example": 80
                },
                "n": {
                  "type": "number",
                  "example": 2
                }
              }
            }
          },
          "required": [
            "model",
            "input",
            "parameters"
          ]
        },
        "WordArtGenerationResponse": {
          "type": "object",
          "properties": {
            "output": {
              "type": "string",
              "description": "Generated word art image URL or data."
            }
          }
        }
      },
      "securitySchemes": {
        "ApiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "Authorization"
        }
      }
    }
  }
"""
    if is_json(schema):
        print('输入字符串schema是JSON')
    elif is_yaml(schema):
        print('输入字符串schema是YAML')
        yaml_dict = yaml.safe_load(schema)
        # 将YAML数据转换为JSON数据
        schema = json.dumps(yaml_dict, indent=2)
    else:
        raise ('输入字符串schema既不是JSON也不是YAML')
    schema_data = json.loads(schema)
    config_data = openapi_schema_convert(schema_data, token)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 检查文件是否为空
        if os.path.getsize(file_path) > 0:
            # 如果文件不为空，读取原有的 JSON 数据
            existing_data = {}
            with open(file_path, 'r', encoding='utf-8') as config_file:
                existing_data = json.load(config_file)
        else:
            # 如果文件为空，创建一个空的字典作为原有数据
            existing_data = {}
    else:
        # 如果文件不存在，创建一个空的字典作为原有数据
        existing_data = {}
        # 将合并后的数据写回 JSON 文件
    existing_data.update(config_data)
    with open(file_path, 'w', encoding='utf-8') as config_file:
        json.dump(existing_data, config_file, ensure_ascii=False, indent=4)


def test_openapi_tool_remote_call():
    DEFAULT_TOOL_CONFIG_FILE = '../../apps/agentfabric/config/additional_tool_config.json'
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)
    tool_cfg = Config.from_file(tool_cfg_file)
    tool = OpenAPISchemaTool(
        cfg=tool_cfg, name='Generate_WordArt_Semantically')
    mock_kwargs = {
        'input.text': '文字创意',
        'input.prompt': '水果，蔬菜，温暖的色彩空间',
        'parameters.steps': 80,
        'parameters.n': 2
    }

    # 调用远程请求，并传递模拟的 kwargs
    try:
        result = tool(remote=True, **mock_kwargs)
        print(result)
        result_data = json.loads(json.dumps(result['result']))
        if 'task_id' in result_data['output']:
            task_id = result_data['output']['task_id']
            print('www', task_id)
            # 从前端重新读取schema字符串
            schema = """
{
    "openapi": "3.1.0",
    "info": {
      "title": "WordArt Semantic Generation API",
      "description": "API for generating semantic word art with customizable parameters.",
      "version": "v1.0.0"
    },
    "servers": [
      {
        "url": "https://dashscope.aliyuncs.com"
      }
    ],
    "paths": {
      "/api/v1/services/aigc/wordart/semantic": {
        "post": {
          "summary": "Generate WordArt Semantically",
          "operationId": "generateWordArt",
          "tags": [
            "WordArt Generation"
          ],
          "requestBody": {
            "required": true,
            "X-DashScope-Async": "enable",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/WordArtGenerationRequest"
                }
              }
            }
          },
          "responses": {
            "200": {
              "description": "Successful Response",
              "content": {
                "application/json": {
                  "schema": {
                    "$ref": "#/components/schemas/WordArtGenerationResponse"
                  }
                }
              }
            }
          },
          "security": [
            {
              "BearerAuth": []
            }
          ]
        }
      },
      "/api/v1/tasks/{task_id}": {
        "get": {
          "summary": "Get WordArt Result",
          "operationId": "getwordartresult",
          "tags": [
            "Get Result"
          ],
          "parameters": [],
          "security": [
            {
              "BearerAuth": []
            }
          ]

        }
      }
    },
    "components": {
      "schemas": {
        "WordArtGenerationRequest": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "enum": ["wordart-semantic"]
            },
            "input": {
              "type": "object",
              "properties":{
                "text": {
                    "type": "string",
                    "example": "文字创意"
                  },
                  "prompt": {
                    "type": "string",
                    "example": "水果，蔬菜，温暖的色彩空间"
                  }
              }
            },
            "parameters": {
              "type": "object",
              "properties": {
                "steps": {
                  "type": "integer",
                  "example": 80
                },
                "n": {
                  "type": "number",
                  "example": 2
                }
              }
            }
          },
          "required": [
            "model",
            "input",
            "parameters"
          ]
        },
        "WordArtGenerationResponse": {
          "type": "object",
          "properties": {
            "output": {
              "type": "string",
              "description": "Generated word art image URL or data."
            }
          }
        }
      },
      "securitySchemes": {
        "ApiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "Authorization"
        }
      }
    }
  }
"""
            schema = schema.replace('{task_id}', task_id)
            schema_data = json.loads(schema)
            existing_data = {}
            # 从前端get token
            config_data = openapi_schema_convert(
                schema_data, 'sk-daee58a99bb44a94bddca4ebb5f6544f')
            with open(file_path, 'r', encoding='utf-8') as config_file:
                existing_data = json.load(config_file)
            existing_data.update(config_data)
            print(existing_data)
            with open(file_path, 'w', encoding='utf-8') as config_file:
                json.dump(
                    existing_data, config_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print('Error:', str(e))
    DEFAULT_TOOL_CONFIG_FILE = '../../apps/agentfabric/config/additional_tool_config.json'
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)
    tool_cfg = Config.from_file(tool_cfg_file)
    tool = OpenAPISchemaTool(cfg=tool_cfg, name='Get_WordArt_Result')
    mock_kwargs = {}
    # 调用远程请求，并传递模拟的 kwargs
    try:
        result = tool(remote=True, **mock_kwargs)
        print(result)
        while True:
            result_data = result.get('result', {})
            output = result_data.get('output', {})
            task_status = output.get('task_status', '')

            if task_status == 'SUCCEEDED':
                print('任务已完成')
                break
            elif task_status == 'FAILED':
                print('任务失败')
                break

            # 继续轮询，等待一段时间后再次调用
            time.sleep(5)  # 等待 5 秒钟
            result = tool(remote=True, **mock_kwargs)
            print(result)
        if 'task_id' in result_data['output'] and 'task_id' in schema:
            task_id = result_data['output']['task_id']
            print('www', task_id)
            # 从前端重新读取schema字符串
            schema = """
{
    "openapi": "3.1.0",
    "info": {
      "title": "WordArt Semantic Generation API",
      "description": "API for generating semantic word art with customizable parameters.",
      "version": "v1.0.0"
    },
    "servers": [
      {
        "url": "https://dashscope.aliyuncs.com"
      }
    ],
    "paths": {
      "/api/v1/services/aigc/wordart/semantic": {
        "post": {
          "summary": "Generate WordArt Semantically",
          "operationId": "generateWordArt",
          "tags": [
            "WordArt Generation"
          ],
          "requestBody": {
            "required": true,
            "X-DashScope-Async": "enable",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/WordArtGenerationRequest"
                }
              }
            }
          },
          "responses": {
            "200": {
              "description": "Successful Response",
              "content": {
                "application/json": {
                  "schema": {
                    "$ref": "#/components/schemas/WordArtGenerationResponse"
                  }
                }
              }
            }
          },
          "security": [
            {
              "BearerAuth": []
            }
          ]
        }
      },
      "/api/v1/tasks/{task_id}": {
        "get": {
          "summary": "Get WordArt Result",
          "operationId": "getwordartresult",
          "tags": [
            "Get Result"
          ],
          "parameters": [],
          "security": [
            {
              "BearerAuth": []
            }
          ]

        }
      }
    },
    "components": {
      "schemas": {
        "WordArtGenerationRequest": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "enum": ["wordart-semantic"]
            },
            "input": {
              "type": "object",
              "properties":{
                "text": {
                    "type": "string",
                    "example": "文字创意"
                  },
                  "prompt": {
                    "type": "string",
                    "example": "水果，蔬菜，温暖的色彩空间"
                  }
              }
            },
            "parameters": {
              "type": "object",
              "properties": {
                "steps": {
                  "type": "integer",
                  "example": 80
                },
                "n": {
                  "type": "number",
                  "example": 2
                }
              }
            }
          },
          "required": [
            "model",
            "input",
            "parameters"
          ]
        },
        "WordArtGenerationResponse": {
          "type": "object",
          "properties": {
            "output": {
              "type": "string",
              "description": "Generated word art image URL or data."
            }
          }
        }
      },
      "securitySchemes": {
        "ApiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "Authorization"
        }
      }
    }
  }
"""
            schema = schema.replace('{task_id}', task_id)
            schema_data = json.loads(schema)
            existing_data = {}
            # 从前端get token
            config_data = openapi_schema_convert(
                schema_data, 'sk-daee58a99bb44a94bddca4ebb5f6544f')
            with open(file_path, 'r', encoding='utf-8') as config_file:
                existing_data = json.load(config_file)
            existing_data.update(config_data)
            with open(file_path, 'w', encoding='utf-8') as config_file:
                json.dump(
                    existing_data, config_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print('Error:', str(e))


if __name__ == '__main__':
    # 从环境获取token，如果这个api不需要token，从前端判断，设置为''
    test_openapi_schema_convert(token='xxxxxxxxxxx')
    test_openapi_tool_remote_call()
