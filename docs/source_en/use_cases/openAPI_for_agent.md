# Extending Agent Capabilities Using API Integration
API (Application Programming Interface) is a set of important protocols, rules, and toolsets that define how software applications interact effectively. When APIs follow the OpenAPI specification, they can be easily called by agents. This integration not only endows agents with more functionalities and capabilities but also greatly expands their range of services.

**For Example:**
1. Accessing Third-Party Services: Through API integration, agents can access various third-party services like weather forecasts, stock quotes, and maps navigation, thereby expanding their range of functionalities.
2. Integrating Third-Party Applications: Through API integration, agents can integrate with third-party applications to achieve more functionalities. For instance, by integrating with payment applications like WeChat and Alipay, agents can facilitate quick payments.
3. Processing Complex Data: API integration allows agents to access more data, enabling them to handle complex data. For example, by integrating news APIs, agents can automatically fetch, analyze, and recommend news information.
4. Automating Processes: Through API integration, agents can automate processes, such as completing transfers automatically by integrating with bank APIs.
5. Personalized Recommendations: With API integration, agents can obtain user preferences and historical behavior data, thereby providing more accurate personalized recommendations. For example, by integrating e-commerce APIs, agents can recommend personalized products.

In summary, **API allows agents to do more!**

## Principle of Agent+API Implementation
The process of an agent using an API includes the following steps:
1. Parameter Analysis: Analyze the user's input JSON or YAML file that complies with the OpenAPI specification to extract key information, including parameters and descriptions;
2. LLM Planning: Fill the extracted information into prompts, submit it to the LLM (Large Language Model), and receive its output;
3. Action Parsing: Analyze the LLM's output. If the LLM indicates that an API call is required, extract the corresponding action information from the output;
4. API Call: Execute an HTTP request based on the parameters in the action information and retrieve the result;
5. LLM Generation: Send the result of the API call back to the LLM to generate the final output content.

## Case Study of Agent Calling API: Word Art Generation
Word Art API: [Click to View](https://help.aliyun.com/zh/dashscope/developer-reference/wordart-quick-start?spm=a2c4g.11186623.0.0.4796b08azcgSVShttps://help.aliyun.com/zh/dashscope/developer-reference/wordart-quick-start?spm=a2c4g.11186623.0.0.4796b08azcgSVS)
Word Art Agent Experience: [Online Experience](https://www.modelscope.cn/studios/Cherrytest/wordartAI/summary)

1. Basic Setup:
   ![Basic Setup](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_fe922135920444c59542cb9c6fe2f848.png)
```
Call the API to generate word art, guiding the user to provide the necessary parameters in the following format:
Text:
Text Material:
Text Background:
Default font used: fangzhengheiti
After generating the task, execute and view the task to display the generated image.
```

2. Schema Configuration:
   ![Schema Configuration](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_16e67a4577954aa98d19a4bbdd895ae7.png)
Explanation of OpenAPI schema writing, based on the API details:

```json
# Example: wordtext_texture
{
    "openapi": "3.1.0",
    "info": {
      "title": "WordArt Texture Generation API", # Custom API title
      "description": "API for generating textured word art with customizable parameters.", # Custom API description
      "version": "v1.0.0"
    },
    "servers": [
      {
        "url": "https://dashscope.aliyuncs.com" # Domain name before the API URL
      }
    ],
    "paths": {
      "/api/v1/services/aigc/wordart/texture": # Part of the API URL excluding the domain name
        "post": # 'post' is the request method, commonly also using 'get'
          "summary": "Generate Textured WordArt", # Custom API summary description, will be used as tool config name
          "operationId": "generate_textured_WordArt", # Custom API operationId, will be used as API tool name
          "tags": [
            "WordArt Generation"
          ], # Custom tags
          "requestBody": # Required field, input information
            "required": true,
            "X-DashScope-Async": "enable", # Keep asynchronous header, written in this location
            "content": {
              "application/json": # content_type
                "schema": {
                  "$ref": "#/components/schemas/WordArtGenerationRequest" # All specific parameters are written under /components/schemas/, 'WordArtGenerationRequest' field is custom, it should match up and down
                }
              }
            }
          },
          "responses": # Output information
            "200": {
              "description": "Successful Response",
              "content": {
                "application/json": {
                  "schema": {
                    "$ref": "#/components/schemas/WordArtGenerationResponse" # Same logic as input
                  }
                }
              }
            }
          },
          "security": [
            {
              "BearerAuth": [] # Default, if there is no API authentication, you can omit this section
            }
          ]
        }
      },
      "/api/v1/tasks/{task_id}": # Other paths under the same domain, note that paths remain at the same hierarchical level
        "get": # 'get' request
          "summary": "Get WordArt Result",
          "operationId": "getwordartresult",
          "tags": [
            "Get Result"
          ],
          "parameters": # 'get' method differs from 'post'; input is not written as requestBody but as tuple-formatted parameters containing dictionaries
          {
            "name":"task_id",
            "in":"path",
            "required":true,
            "description":"The unique identifier of the word art generation task",
            "schema":{
              "type":"string"
            }
          }
          ],
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
        "WordArtGenerationRequest": # Specific input parameters
          "type": "object", # Default
          "properties": # Default, includes api input information such as type, enum (optional values), example, description, etc.
            "model": {
              "type": "string",
              "enum": ["wordart-texture"]
            },
            "input": { # For nested object structures like 'input', maintain the structure, and write the required parameters in input["properties"]
              "type": "object",
              "properties":{
                "text": { # 'text' is also a nested object
                    "type": "object",
                    "properties": {
                      "text_content": {
                      "type": "string",
                      "example": "文字纹理",
                      "description": "The text that the user wants to convert to word art",
                      "required":true
                      },
                      "font_name": {
                      "type": "string",
                      "example": "dongfangdakai",
                      "description": "The font format in which the user wants to convert the text to word art",
                      "required":true
                      }
                    }
                  },
                  "prompt": {
                    "type": "string",
                    "example": "水果，蔬菜，温暖的色彩空间",
                    "description": "The user's style requirements for the word art, which may include shape, color, entity, etc.",
                    "required":true
                  }
              }
            },
            "parameters": # Nested parameters
              "type": "object",
              "properties": {
                "n": {
                  "type": "number",
                  "example": 2
                }
              }
            }
          },
# Required parameters. If nested, write only the outermost layer (e.g., input, parameters). All nested parameters will be set to required.
If you want to change default values that are not required, you can add them above and include their names in the required array. If some parameters are not required and you only want to keep their default values, you can write them like model, input, etc. above, and omit their names from the required array (though this might not serve much purpose).
          "required": [
            "model",
            "input",
            "parameters"
          ]
        },
        "WordArtGenerationResponse": # Specific output parameters. Can be roughly modeled like the input, or written out in detail similar to the 'request', without the required field.
          "type": "object",
          "properties": {
            "output": {
              "type": "string",
              "description": "Generated word art image URL or data."
            }
          }
        }
      },
      "securitySchemes": # You can omit this section if there is no API authentication
        "ApiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "Authorization"
        }
      }
    }
  }
```
Directly executable code:
```json
{
    "openapi":"3.1.0",
    "info":{
        "title":"WordArt Texture Generation API",
        "description":"API for generating textured word art with customizable parameters.",
        "version":"v1.0.0"
    },
    "servers":[
        {
            "url":"https://dashscope.aliyuncs.com"
        }
    ],
    "paths":{
        "/api/v1/services/aigc/wordart/texture":{
            "post":{
                "summary":"Generate Textured WordArt",
                "operationId":"generate_textured_WordArt",
                "tags":[
                    "WordArt Generation"
                ],
                "requestBody":{
                    "required":true,
                    "X-DashScope-Async":"enable",
                    "content":{
                        "application/json":{
                            "schema":{
                                "$ref":"#/components/schemas/WordArtGenerationRequest"
                            }
                        }
                    }
                },
                "responses":{
                    "200":{
                        "description":"Successful Response",
                        "content":{
                            "application/json":{
                                "schema":{
                                    "$ref":"#/components/schemas/WordArtGenerationResponse"
                                }
                            }
                        }
                    }
                },
                "security":[
                    {
                        "BearerAuth":[

                        ]
                    }
                ]
            }
        },
        "/api/v1/tasks/{task_id}":{
            "get":{
                "summary":"Get WordArt Result",
                "operationId":"getwordartresult",
                "tags":[
                    "Get Result"
                ],
                "parameters":[
                    {
                        "name":"task_id",
                        "in":"path",
                        "required":true,
                        "description":"The unique identifier of the word art generation task",
                        "schema":{
                            "type":"string"
                        }
                    }
                ],
                "security":[
                    {
                        "BearerAuth":[

                        ]
                    }
                ]
            }
        }
    },
    "components":{
        "schemas":{
            "WordArtGenerationRequest":{
                "type":"object",
                "properties":{
                    "model":{
                        "type":"string",
                        "enum":[
                            "wordart-texture"
                        ]
                    },
                    "input":{
                        "type":"object",
                        "properties":{
                            "text":{
                                "type":"object",
                                "properties":{
                                    "text_content":{
                                        "type":"string",
                                        "example":"文字纹理",
                                        "description":"用户想要转为艺术字的文本",
                                        "required":true
                                    },
                                    "font_name":{
                                        "type":"string",
                                        "example":"dongfangdakai",
                                        "description":"用户想要转为艺术字的字体格式，如果用户没有提供，就传入默认值dongfangdakai",
                                        "required":true,
                                        "enum":[
                                            "cangeryuyangti_b",
                                            "siyuansongti_h",
                                            "puhuiti_m",
                                            "fangzhengheiti",
                                            "siyuansongti_b",
                                            "kuaileti",
                                            "jinbuti",
                                            "fangzhengkaiti",
                                            "fangzhengfangsong",
                                            "siyuanheiti_l",
                                            "cangeryuyangti_r",
                                            "gufeng_2",
                                            "gufeng_1",
                                            "siyuanheiti_m",
                                            "cangeryuyangti_h",
                                            "kuheiti",
                                            "logoti",
                                            "cangeryuyangti_l",
                                            "fangzhengshusong",
                                            "siyuanheiti_b",
                                            "wenyiti",
                                            "siyuanheiti_h",
                                            "siyuansongti_m",
                                            "siyuansongti_r",
                                            "shuheiti",
                                            "cangeryuyangti_m",
                                            "puhuiti",
                                            "dongfangdakai",
                                            "siyuanheiti_r",
                                            "puhuiti_l",
                                            "siyuansongti_l",
                                            "gufeng_3"
                                        ]
                                    }
                                }
                            },
                            "prompt":{
                                "type":"string",
                                "example":"水果，蔬菜，温暖的色彩空间",
                                "description":"用户对艺术字的风格要求，可能是形状、颜色、实体等方面的要求",
                                "required":true
                            }
                        }
                    },
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "n":{
                                "type":"number",
                                "example":2,
                                "description":"取值范围为1-4的整数",
                                "required":true
                            }
                        }
                    }
                },
                "required":[
                    "model",
                    "input",
                    "parameters"
                ]
            },
            "WordArtGenerationResponse":{
                "type":"object",
                "properties":{
                    "output":{
                        "type":"string",
                        "description":"Generated word art image URL or data."
                    }
                }
            }
        },
        "securitySchemes":{
            "ApiKeyAuth":{
                "type":"apiKey",
                "in":"header",
                "name":"Authorization"
            }
        }
    }
}
```

3. Agent Running Effect
   ![Agent Running Effect](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_da364c9d264b4d9ab1c0b46ac53d10ea.png)
4. Word Art Generation Effect
   ![Word Art Generation Effect](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_dcbd5a8f08f747fc8877213a83f8ee0a.png)
