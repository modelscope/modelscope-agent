support_lang = ['zh-cn', 'en']

i18n = {
    'create': ['创建', 'Create'],
    'configure': ['配置', 'Configure'],
    'send': ['发送', 'Send'],
    'sendOnLoading': ['发送（Agent 加载中...）', 'Send (Agent Loading...)'],
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
    'build_hint': ['点击"构建"完成构建', 'Click "Build" to finish building'],
    'publish_hint': [
        '点击"发布"跳转创空间完成 Agent 发布',
        'Click "Publish" to jump to the space to finish agent publishing'
    ]
}


class I18n():

    def __init__(self, lang):
        self.lang = lang
        self.langIndex = support_lang.index(lang)

    def get(self, field):
        return i18n.get(field)[self.langIndex]

    def get_whole(self, field):
        return f'{i18n.get(field)[0]}({i18n.get(field)[1]})'
