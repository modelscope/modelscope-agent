from .tool import Tool


class AliyunRenewInstanceTool(Tool):

    description = '续费一台包年包月ECS实例'
    name = 'RenewInstance'
    parameters: list = [{
        'name': 'instance_id',
        'description': 'ECS实例ID',
        'required': True
    },
    {
        'name': 'period',
        'description': '续费时长以月为单位',
        'required': True
    }
    ]

    def _local_call(self, *args, **kwargs):
        instance_id = kwargs['instance_id']
        period = kwargs['period']
        return {'result': f'已完成ECS实例ID为{instance_id}的续费，续费时长{period}月'}
