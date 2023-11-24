import os
import sys
import json
from modelscope_agent.tools.openapi_tool import (OpenAPISchemaTool,openapi_schema_convert)
from modelscope.utils.config import Config
sys.path.append('../../')
#before run test,add your tokeen
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
    schema_json = '''
  {
    "openapi": "3.1.0",
    "info": {
      "title": "Combined API Services",
      "description": "API services for file uploading and face detection and face training in images.",
      "version": "v1.0.0"
    },
    "servers": [
      {
        "url": "https://dashscope.aliyuncs.com"
      }
    ],
    "paths": {
      "/api/v1/services/vision/facedetection/detect": {
        "post": {
          "summary": "Detect Faces in Images",
          "operationId": "detectFaces",
          "tags": [
            "Face Detection"
          ],
          "requestBody": {
            "required": true,
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FaceDetectionRequest"
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
                    "$ref": "#/components/schemas/FaceDetectionResponse"
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
      }
    },
    "components": {
      "schemas": {
        "FaceDetectionRequest": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "enum": [
                "facechain-facedetect"
              ]
            },
            "input": {
              "type": "object",
              "properties": {
                "images": {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "format": "url"
                  }
                }
              }
            },
            "parameters": {
              "type": "object"
            }
          },
          "required": [
            "model",
            "input"
          ]
        },
        "FaceDetectionResponse": {
          "type": "object",
          "properties": {
            "output": {
              "type": "object",
              "properties": {
                "is_face": {
                  "type": "array",
                  "items": {
                    "type": "boolean",
                    "description": "List of results corresponding to the submitted images.",
                    "example": [true, true, false, false]
                  }
                }
              }
            },
            "request_id": {
              "type": "string",
              "description": "Unique code for the request.",
              "example": "7574ee8f-38a3-4b1e-9280-11c33ab46e51"
            }
          }
        }
      },
      "securitySchemes": {
        "BearerAuth": {
          "type": "http",
          "scheme": "bearer",
          "bearerFormat": "JWT"
        }
      }
    }
  }

  '''
    if is_json(schema):
      print("输入字符串schema是JSON")
    elif is_yaml(schema):
      print("输入字符串schema是YAML")
      yaml_dict = yaml.safe_load(schema)
      # 将YAML数据转换为JSON数据
      schema = json.dumps(yaml_dict, indent=2)
    else:
        raise("输入字符串schema既不是JSON也不是YAML")
    schema_data = json.loads(schema)
    schema = json.loads(schema_data)
    config_data = openapi_schema_convert(schema, YOUR_API_TOKEN=token)

    with open(
            '../../apps/agentfabric/config/additional_tool_config.json',
            'a',
            encoding='utf-8') as config_file:
        json.dump(config_data, config_file, ensure_ascii=False, indent=4)


def test_openapi_tool_remote_call():
    DEFAULT_TOOL_CONFIG_FILE = '../../apps/agentfabric/config/tool_config.json'
    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE', DEFAULT_TOOL_CONFIG_FILE)
    tool_cfg = Config.from_file(tool_cfg_file)
    tool = OpenAPISchemaTool(cfg=tool_cfg, name='Detect Faces in Images')
    restored_dict = {}
    mock_kwargs = {
        'model':
        'facechain-facedetect',
        'input.images': [
            'http://finetune-swap-wulanchabu.oss-cn-wulanchabu.aliyuncs.com/zhicheng/tmp/1E1D5AFA-3C3A-4B6F-ABD6-8742CA983C42.png',
            'http://finetune-swap-wulanchabu.oss-cn-wulanchabu.aliyuncs.com/zhicheng/tmp/3.JPG',
            'http://finetune-swap-wulanchabu.oss-cn-wulanchabu.aliyuncs.com/zhicheng/tmp/F2EA3984-6EE2-44CD-928F-109B7276BCB6.png'
        ]
    }
    for key, value in mock_kwargs.items():
        if '.' in key:
            keys = key.split('.')
            temp_dict = restored_dict
            for k in keys[:-1]:
                temp_dict = temp_dict.setdefault(k, {})
            temp_dict[keys[-1]] = value
        else:
            restored_dict[key] = value
    mock_kwargs = restored_dict
    try:
        result = tool(remote=True, **mock_kwargs)
        print(
            result
        )  #{'result': "{'output': {'is_face': [True, True, True]}, 'usage': {}, 'request_id': '355cca45-b1f2-942e-8f55-a63c3b34fdf3'}"}
    except Exception as e:
        print('Error:', str(e))

if __name__ == '__main__':
    test_openapi_schema_convert(token='xxxxxxxx')
    test_openapi_tool_remote_call()
