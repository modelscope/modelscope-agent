from modelscope_agent.tools.base import BaseTool, register_tool


@register_tool('RenewInstance')
class AliyunRenewInstanceTool(BaseTool):
    description = '续费一台包年包月ECS实例'
    name = 'RenewInstance'
    parameters: list = [{
        'name': 'instance_id',
        'description': 'ECS实例ID',
        'required': True,
        'type': 'string'
    }, {
        'name': 'period',
        'description': '续费时长以月为单位',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs):
        params = self._verify_args(params)
        instance_id = params['instance_id']
        period = params['period']
        return str({'result': f'已完成ECS实例ID为{instance_id}的续费，续费时长{period}月'})
