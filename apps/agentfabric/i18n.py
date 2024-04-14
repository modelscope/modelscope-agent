# flake8: noqa
support_lang = ['zh-cn', 'en']

i18n = {
    'create': ['创建', 'Create'],
    'configure': ['配置', 'Configure'],
    'send': ['发送', 'Send'],
    'sendOnLoading': ['发送（Agent 加载中...）', 'Send (Agent Loading...)'],
    'upload_btn': ['上传文件', 'Upload File'],
    'message': ['输入', 'Send a message'],
    'message_placeholder': ['输入你的消息', 'Type your message here'],
    'prompt_suggestion': ['推荐提示词', 'Prompt Suggestions'],
    'form_avatar': ['头像', 'Avatar'],
    'form_name': ['名称', 'Name'],
    'form_name_placeholder': ['为你的 agent 取一个名字', 'Name your agent'],
    'form_description': ['描述', 'Description'],
    'form_description_placeholder': [
        '为你的 agent 添加一段简短的描述',
        'Add a short description about what this agent does'
    ],
    'form_instructions': ['指令', 'Instructions'],
    'form_instructions_placeholder': [
        '你的 agent 需要处理哪些事情',
        'What does this agent do? How does it behave? What should it avoid doing?'
    ],
    'form_model': ['模型', 'Model'],
    'form_agent_language': ['Agent 语言', 'Agent Language'],
    'form_prologue': ['开场白', 'Prologue'],
    'form_prologue_placeholder':
    ['为你的 agent 添加一个开场白', 'Add an opening line to your agent'],
    'form_prompt_suggestion':
    ['推荐提示词，双击行可修改', 'agents suggestion，double click to modify'],
    'form_knowledge': ['知识库', 'Knowledge Base'],
    'form_knowledge_upload_button':
    ['添加文件到知识库', 'Add files to the knowledge base'],
    'form_capabilities': ['内置能力', 'Capabilities'],
    'form_update_button': ['更新配置', 'Update Configuration'],
    'open_api_accordion': ['OpenAPI 配置', 'OpenAPI Configuration'],
    'preview': ['预览', 'Preview'],
    'build': ['构建', 'Build'],
    'publish': ['发布', 'Publish'],
    'import_config': ['导入配置', 'Import Config'],
    'space_addr': ['你的AGENT_URL', 'Yours AGENT_URL'],
    'input_space_addr': ['输入你的AGENT_URL', 'input your agent_url here'],
    'import_space': ['导入你的Agent', 'Import your existing agent'],
    'import_hint': [
        '输入你创空间环境变量AGENT_URL，点击导入配置',
        'input your AGNET_URL which lies in your env of your space, then type Import Config'
    ],
    'build_hint': ['点击"构建"完成构建', 'Click "Build" to finish building'],
    'publish_hint': [
        '点击"发布"跳转创空间完成 Agent 发布',
        'Click "Publish" to jump to the space to finish agent publishing'
    ],
    'publish_alert': [
        """#### 注意：Agent实际发布时需要配置相关API的key。
- 千问、万相、艺术字等 DashScope API 所需： [申请入口](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)
- 高德天气 API： [申请入口](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather)
- Web Searching API： [申请入口](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
- ModelScope 语音生成、视频生成 API：[申请入口](https://modelscope.cn/my/myaccesstoken)""",
        """#### Note: The key of the relevant API needs to be configured when the Agent is actually released.
- Qwen,Wanx,WordArt,etc DashScope API: [Application entrance](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)
- Amap Weather API: [Application entrance](https://lbs.amap.com/api/javascript-api-v2/guide/services/weather)
- Web Searching API： [Application entrance](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
- ModelScope speech-generation,video-generation API: [Application entrance](https://modelscope.cn/my/myaccesstoken)
"""
    ],
    'header': [
        '<span style="font-size: 20px; font-weight: 500;">\N{fire} AgentFabric -- 由 Modelscope-agent 驱动 </span> [github 点赞](https://github.com/modelscope/modelscope-agent/tree/main)',  # noqa E501
        '<span style="font-size: 20px; font-weight: 500;">\N{fire} AgentFabric powered by Modelscope-agent </span> [github star](https://github.com/modelscope/modelscope-agent/tree/main)'  # noqa E501
    ],
}


class I18n():

    def __init__(self, lang):
        self.lang = lang
        self.langIndex = support_lang.index(lang)

    def get(self, field):
        return i18n.get(field)[self.langIndex]

    def get_whole(self, field):
        return f'{i18n.get(field)[0]}({i18n.get(field)[1]})'
