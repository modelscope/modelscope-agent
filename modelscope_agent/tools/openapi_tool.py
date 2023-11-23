import sys
sys.path.append("../../")
import os
import json
from jsonschema import RefResolver
from typing import List, Optional
from pydantic import BaseModel, ValidationError
import requests
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3

class ParametersSchema(BaseModel):
    name: str
    description: str
    required: Optional[bool] = True


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: List[ParametersSchema]

class OpenAPISchemaTool():
    """
     openai schema tool
    """
    name: str = 'api tool'
    description: str = 'This is a api tool that ...'
    parameters: list = []
    
    def __init__(self, cfg, name):
        self.name = name
        self.cfg = cfg.get(self.name, {})
        self.is_remote_tool = self.cfg.get('is_remote_tool', False)

        # remote call
        self.url = self.cfg.get('url', '')
        self.token = self.cfg.get('token', '')
        self.header = self.cfg.get('header','')
        self.method = self.cfg.get('method','')
        self.parameters = self.cfg.get('parameters',[])
        self.description = self.cfg.get('description','This is a api tool that ...')
        self.responses_param = self.cfg.get('responses_param',[])
        try:
            all_para = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_para)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')
        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(all_para)

    def __call__(self, remote=False, *args, **kwargs):
        if self.is_remote_tool or remote:
            return self._remote_call(*args, **kwargs)
        else:
            return self._local_call(*args, **kwargs)
    
    def _remote_call(self, *args, **kwargs):
        if self.url == '':
            raise ValueError(
                f"Could not use remote call for {self.name} since this tool doesn't have a remote endpoint"
            )

        remote_parsed_input = json.dumps(
            self._remote_parse_input(*args, **kwargs))

        origin_result = None
        if self.method == "POST":
            retry_times = MAX_RETRY_TIMES
            while retry_times:
                retry_times -= 1
                try:
                    response = requests.request(
                        'POST',
                        url = self.url,
                        headers = self.header,
                        data = remote_parsed_input)
                    if response.status_code != requests.codes.ok:
                            response.raise_for_status()

                    origin_result = str(json.loads(
                        response.content.decode('utf-8')))

                    final_result = self._parse_output(origin_result, remote=True)
                    return final_result
                except Timeout:
                    continue
                except RequestException as e:
                    raise ValueError(
                        f'Remote call failed with error code: {e.response.status_code},\
                        error message: {e.response.content.decode("utf-8")}')

            raise ValueError(
                'Remote call max retry times exceeded! Please try to use local call.'
            )
        elif self.method == "GET":
            retry_times = MAX_RETRY_TIMES
            while retry_times:
                retry_times -= 1
                try:
                    response = requests.request(
                        'GET',
                        url = self.url,
                        headers = self.header,
                        data = remote_parsed_input)
                    if response.status_code != requests.codes.ok:
                            response.raise_for_status()

                    origin_result = str(json.loads(
                        response.content.decode('utf-8')))

                    final_result = self._parse_output(origin_result, remote=True)
                    return final_result
                except Timeout:
                    continue
                except RequestException as e:
                    raise ValueError(
                        f'Remote call failed with error code: {e.response.status_code},\
                        error message: {e.response.content.decode("utf-8")}')

            raise ValueError(
                'Remote call max retry times exceeded! Please try to use local call.'
            )
        else:
            raise ValueError('Remote call method is invalid!We have POST and GET method.')
    def _local_call(self, *args, **kwargs):
        return 
    
    def _remote_parse_input(self,*args, **kwargs):
        model_name = kwargs.pop('model', '')
        if model_name == '':
          model_name = self.parameters.get('value')[0]
        kwargs['model'] = model_name
        restored_dict = {}
        for key, value in kwargs.items():
            # 检查键中是否包含 "."
            if "." in key:
                # 按 "." 分割键，并创建嵌套字典结构
                keys = key.split(".")
                temp_dict = restored_dict
                for k in keys[:-1]:
                    temp_dict = temp_dict.setdefault(k, {})
                temp_dict[keys[-1]] = value
            else:
                # 如果键中不包含 "."，直接将键值对存入 restored_dict
                restored_dict[key] = value
            kwargs = restored_dict
        return kwargs
    
    def _local_parse_input(self, *args, **kwargs):
        return args, kwargs
    
    def _parse_output(self, origin_result, *args, **kwargs):
        return {'result': origin_result}
    def __str__(self):
        return self._str

    def get_function(self):
        return self._function

    def parse_pydantic_model_to_openai_function(self, all_para: dict):
        '''
        this method used to convert a pydantic model to openai function schema
        such that convert
        all_para = {
            'name': get_current_weather,
            'description': Get the current weather in a given location,
            'parameters': [{
                'name': 'image',
                'description': '用户输入的图片',
                'required': True
            }, {
                'name': 'text',
                'description': '用户输入的文本',
                'required': True
            }]
        }
        to
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "用户输入的图片",
                    },
                    "text": {
                        "type": "string",
                        "description": "用户输入的文本",
                    },
                "required": ["image", "text"],
            },
        }
        '''
        function = {
            'name': all_para['name'],
            'description': all_para['description'],
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        }
        for para in all_para['parameters']:
            function['parameters']['properties'][para['name']] = {
                'type': 'string',
                'description': para['description']
            }
            if para['required']:
                function['parameters']['required'].append(para['name'])

        return function

#openapi_schema_convert,register to tool_config.json
def extract_references(schema_content):
    references = []
    if isinstance(schema_content, dict):
        if "$ref" in schema_content:
            references.append(schema_content["$ref"])
        for key, value in schema_content.items():
            references.extend(extract_references(value))
    elif isinstance(schema_content, list):
        for item in schema_content:
            references.extend(extract_references(item))
    return references

def parse_nested_parameters(param_name, param_info, parameters_list,content):
    param_type = param_info["type"]
    param_description = param_info.get("description", f"用户输入的{param_name}")  # 按需更改描述
    param_required = param_name in content["required"]
    try:
      if param_type == "object":
          properties = param_info.get("properties")
          if properties:
              #print(properties)
              # 如果参数类型是对象且具有非空的 "properties" 字段，则递归解析其内部属性
              for inner_param_name, inner_param_info in properties.items():
                  param_type = inner_param_info["type"]
                  param_description = inner_param_info.get("description", f"用户输入的{param_name}.{inner_param_name}")  
                  param_required = param_name in content["required"]  
                  parameters_list.append({
                      "name": f"{param_name}.{inner_param_name}",
                      "description": param_description,
                      "required": param_required,
                      "type": param_type,
                      "value": inner_param_info.get("enum", "")
          })
      else:
          # 非嵌套的参数，直接添加到参数列表
          parameters_list.append({
              "name": param_name,
              "description": param_description,
              "required": param_required,
              "type": param_type,
              "value": param_info.get("enum", "")
          })
    except Exception as e:
      raise ValueError(f"{e}:schema结构出错" )
def parse_responses_parameters(param_name, param_info, parameters_list):
    param_type = param_info["type"]
    param_description = param_info.get("description", f"调用api返回的{param_name}")  # 按需更改描述
    try:
      if param_type == "object":
          properties = param_info.get("properties")
          if properties:
              #print(properties)
              # 如果参数类型是对象且具有非空的 "properties" 字段，则递归解析其内部属性
              for inner_param_name, inner_param_info in properties.items():
                  param_type = inner_param_info["type"]
                  param_description = inner_param_info.get("description", f"调用api返回的{param_name}.{inner_param_name}")  
                  
                  parameters_list.append({
                      "name": f"{param_name}.{inner_param_name}",
                      "description": param_description, 
                      "type": param_type,
          })
      else:
          # 非嵌套的参数，直接添加到参数列表
          parameters_list.append({
              "name": param_name,
              "description": param_description,
              "type": param_type, 
          })
    except Exception as e:
      raise ValueError(f"{e}:schema结构出错" )
    
#YOUR_API_TOKEN = os.getenv('YOUR_API_TOKEN') ## get token from gradio
 #schema is json data
def openapi_schema_convert(schema,YOUR_API_TOKEN):
  resolver = RefResolver.from_schema(schema)
  servers = schema.get('servers', [])
  if servers:
      servers_url = servers[0].get('url')
  else:
      print("No URL found in the schema.")
  # Extract endpoints
  endpoints = schema.get('paths',{})
  description = schema.get("info",{}).get('description','This is a api tool that ...')

  # 定义一个空的配置字典
  config_data = {}

  # 遍历每个端点和其内容
  for endpoint_path, methods in endpoints.items():
      for method, details in methods.items():
          summary = details.get('summary', 'No summary')
          name = details.get('operationId', 'No operationId')
          url = f'{servers_url}{endpoint_path}'
          requestBody = details.get('requestBody', {})
          responses = details.get('responses', {})['200']
          security = details.get('security', [{}])
          # Security (Bearer Token)
          if security:
              for sec in security:
                  if "BearerAuth" in sec:
                      print("Requires Bearer Token Authentication")
                      authorization = f'Bearer {YOUR_API_TOKEN}'#  YOUR_API_TOKEN 替换为实际的令牌
          if requestBody:
              for content_type, content_details in requestBody.get('content', {}).items():
                  schema_content = content_details.get("schema", {})        
                  references = extract_references(schema_content)
                  for reference in references:
                      resolved_schema = resolver.resolve(reference)
                      content = resolved_schema[1]
                      parameters_list = []
                        
                      for param_name, param_info in content["properties"].items():
                          parse_nested_parameters(param_name, param_info, parameters_list,content)
                  
          if responses:   
              for content_type, content_details in responses.get('content', {}).items():
                  schema_content = content_details.get("schema", {})        
                  references = extract_references(schema_content)
                  for reference in references:
                      resolved_schema = resolver.resolve(reference)
                      content = resolved_schema[1]
                      responses_list = []
                        
                      for param_name, param_info in content["properties"].items():
                          parse_responses_parameters(param_name, param_info, responses_list)
                      
                      config_entry = {
                                  "name": name,
                                  "description": description,
                                  "is_active": True,
                                  "is_remote_tool": True,
                                  "url": url,
                                  "method": method.upper(),
                                  "parameters": parameters_list,
                                  "header": {
                                      'Content-Type': content_type,
                                      'Authorization': authorization  
                                  },
                                  "responses_param": responses_list
                              }
                          # 将配置添加到配置字典中
                      config_data[summary] = config_entry
                      print(config_data[summary])
                  # 将配置字典写入 cfg.json 文件
                      with open('../../apps/agentfabric/config/tool_config.json', 'r',encoding='utf-8') as config_file:
                          existing_data = json.load(config_file)
                      existing_data.update(config_data)
                      with open('../../apps/agentfabric/config/tool_config.json', 'w', encoding='utf-8') as config_file:
                          json.dump(existing_data, config_file, ensure_ascii=False, indent=4) 
                                               
  print("Schema Configuration has been saved to tool_config.json")


