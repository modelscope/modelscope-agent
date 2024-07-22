# openAPI接入来扩展Agent的能力范围

API（Application Programming Interface）是一组重要的协议、规则和工具集，它们定义了软件应用程序之间如何进行有效的交互。当API遵循OpenAPI规范时，它们便可以被agent轻松调用。这种接入不仅赋予agent更多的功能和能力，而且可以极大地扩展其服务范围。

**例如：**

1. 访问第三方服务：通过API接入，Agent可以访问各种第三方服务，如天气预报、股票行情、地图导航等，从而扩展自己的功能范围。
2. 集成第三方应用：通过API接入，Agent可以与第三方应用进行集成，实现更多的功能，例如通过集成微信、支付宝等支付应用，实现快捷支付。
3. 处理复杂数据：通过API接入，Agent可以获得更多的数据，从而处理更复杂的数据。例如通过接入新闻API，实现自动获取新闻信息，然后进行分析和推荐。
4. 自动化流程：通过API接入，Agent可以实现自动化流程，例如通过接入银行的API，实现自动完成转账等操作。
5. 个性化推荐：通过API接入，Agent可以获取用户的偏好和历史行为数据，从而实现更准确的个性化推荐。例如通过接入电商API，实现个性化商品推荐。

总结就是<u>**API可以让Agent做到更多的事情**</u>！

## Agent+API实现原理

Agent使用API的流程包括以下几个步骤：
1. 参数分析：分析用户输入的符合OpenAPI规范的JSON或YAML文件，提取关键信息，包括参数和描述等；
2. LLM Planning：将提取的信息填写到提示语中，提交给LLM (大型语言模型) 并接收其输出；
3. 动作解析：分析LLM的输出，如果LLM指示需要调用API，则从输出中提取相应的动作信息；
4. API 调用：根据动作信息中的参数，执行HTTP请求，获取结果；
5. LLM Generation：将API调用的结果传回LLM，获取最终的输出内容。

## Agent调用API案例：艺术字生成

艺术字API：[点击跳转](https://help.aliyun.com/zh/dashscope/developer-reference/wordart-quick-start?spm=a2c4g.11186623.0.0.4796b08azcgSVShttps://help.aliyun.com/zh/dashscope/developer-reference/wordart-quick-start?spm=a2c4g.11186623.0.0.4796b08azcgSVS)

艺术字 Agent体验：[在线体验](https://www.modelscope.cn/studios/Cherrytest/wordartAI/summary)

1. 基础设置
   ![基础设置](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_fe922135920444c59542cb9c6fe2f848.png)

```
调用API生成艺术字，引导用户按以下格式提供必要参数
文字：
文字材料：
文字背景：
文字字体默认使用：fangzhengheiti
生成任务后执行查看任务，展示生成的图片
```

2. schema配置
   ![schema配置](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_16e67a4577954aa98d19a4bbdd895ae7.png)

Openapi schema 编写解释，根据api详情进行编写：

```json
#案例：wordtext_texture
{
    "openapi": "3.1.0",
    "info": {
      "title": "WordArt Texture Generation API",#自定义API title
      "description": "API for generating textured word art with customizable parameters.", #自定义api rescription
      "version": "v1.0.0"
    },
    "servers": [
      {
        "url": "https://dashscope.aliyuncs.com"#API url前面的域名
      }
    ],
    "paths": {
      "/api/v1/services/aigc/wordart/texture": #API URL 除去域名的部分{
        "post": #post 是request方法，常用的还有get {
          "summary": "Generate Textured WordArt",#自定义api的总结描述，会作为tool config name
          "operationId": "generate_textured_WordArt",#自定义api operationId，会作为api tool name
          "tags": [
            "WordArt Generation"
          ], #自定义
          "requestBody": #必需字段，入参的信息{
            "required": true,
            "X-DashScope-Async": "enable",#保持异步的header，写在这个位置
            "content": {
              "application/json": #content_type {
                "schema": {
                  "$ref": "#/components/schemas/WordArtGenerationRequest"#将具体的参数都统一写到/components/schemas/下，WordArtGenerationRequest字段自定义，上下一
      致即可
          }
              }
            }
          },
          "responses": #出参信息{
            "200": {
              "description": "Successful Response",
              "content": {
                "application/json": {
                  "schema": {
                    "$ref": "#/components/schemas/WordArtGenerationResponse"#与前面入参同理
                  }
                }
              }
            }
          },
          "security": [
            {
              "BearerAuth": []#默认，没有api鉴权就不写这段
            }
          ]
        }
      },
      "/api/v1/tasks/{task_id}": #同一域名下的其他path，注意path之间保持同一级{
        "get": #get请求{
          "summary": "Get WordArt Result",
          "operationId": "getwordartresult",
          "tags": [
            "Get Result"
          ],
          "parameters": [#get 方式与post不同，入参不需要像上面写requestbody，写成这种parameters格式的元组里包含字典即可
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
        "WordArtGenerationRequest":#入参具体参数信息 {
          "type": "object",#默认
          "properties": #默认，里面是根据api 入参信息的type，enum（可选的值，比如model ），example，description等{
            "model": {
              "type": "string",
              "enum": ["wordart-texture"]
            },
            "input": {#像input这种object嵌套结构，需要保持，再在input["properties"]里面写必需传入的参数
              "type": "object",
              "properties":{
                "text": {#text也是object嵌套
                    "type": "object",
                    "properties": {
                      "text_content": {
                      "type": "string",
                      "example": "文字纹理",
                      "description": "用户想要转为艺术字的文本",
                      "required":true
                      },
                      "font_name": {
                      "type": "string",
                      "example": "dongfangdakai",
                      "description": "用户想要转为艺术字的字体格式",
                      "required":true
                      }
                    }
                  },
                  "prompt": {
                    "type": "string",
                    "example": "水果，蔬菜，温暖的色彩空间",
                    "description": "用户对艺术字的风格要求，可能是形状、颜色、实体等方面的要求",
                    "required":true
                  }
              }
            },
            "parameters": #嵌套{
              "type": "object",
              "properties": {
                "n": {
                  "type": "number",
                  "example": 2
                }
              }
            }
          },
#需要传入的参数，如果有嵌套结构，只需要写最外层（比如input，parameters），此时嵌套里面的参数都会置为required。
当你想改变那些不是必需上传的默认值时，你也可以写到上面，然后在required里写上这个参数名。当有些参数不是必需上传而且你只想保持它的默认值时，你可以把信息像model，input类似写到上面，然后在required这个元组里不写（这其实没啥意义，不如上面的信息就不写 ）
          "required": [
            "model",
            "input",
            "parameters"
          ]
        },
        "WordArtGenerationResponse": #具体的出参信息，可以仿照下面粗略把整个response都放到一起 ，也可以仿照request写出每一个出参，不需要写required字段{
          "type": "object",
          "properties": {
            "output": {
              "type": "string",
              "description": "Generated word art image URL or data."
            }
          }
        }
      },
      "securitySchemes":#没有api鉴权可以不写 {
        "ApiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "Authorization"
        }
      }
    }
  }
```

直接可运行代码：

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

3. Agent运行效果
   ![Agent运行效果](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_da364c9d264b4d9ab1c0b46ac53d10ea.png)

4. 艺术字生成效果
   ![艺术字生成效果](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_dcbd5a8f08f747fc8877213a83f8ee0a.png)
