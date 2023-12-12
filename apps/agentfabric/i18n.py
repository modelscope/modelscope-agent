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
    'form_prompt_suggestion':
    ['推荐提示词，双击行可修改', 'prompt suggestion，double click to modify'],
    'form_knowledge': ['知识库', 'Knowledge Base'],
    'form_capabilities': ['内置能力', 'Capabilities'],
    'form_update_button': ['更新配置', 'Update Configuration'],
    'open_api_accordion': ['OpenAPI 配置', 'OpenAPI Configuration'],
    'preview': ['预览', 'Preview'],
    'build': ['构建', 'Build'],
    'publish': ['发布', 'Publish'],
    'update': ['更新', 'Update'],
    'space_addr': ['你的AGENT_URL', 'Yours AGENT_URL'],
    'input_space_addr': ['输入你的AGENT_URL', 'input your agent_url here'],
    'import_space': ['导入你的Agent', 'Import your existing agent'],
    'import_hint': [
        '输入你创空间环境变量AGENT_URL，点击更新',
        'input your AGNET_URL which lies in your env of your space, then type Update'
    ],
    'build_hint': ['点击"构建"完成构建', 'Click "Build" to finish building'],
    'publish_hint': [
        '点击"发布"跳转创空间完成 Agent 发布',
        'Click "Publish" to jump to the space to finish agent publishing'
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
