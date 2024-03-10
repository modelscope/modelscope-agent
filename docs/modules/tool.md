## Tools

We provide some default pipeline tools of multiple domain that integrates in modelscope.

Also, you can custom your tools by inheriting base tool and define names, descriptions, and parameters according to pre-defined schema. And you can implement `_local_call()` or `_remote_call()` according to your requirement. An example of custom tool is provided below in [custom_tool](../../modelscope_agent/tools/custom_tool.py):

```python
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.tools import register_tool

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

```

You can also refer to our provided demo for a guide on how to use the new tool in [demo_register_new_tool](../../demo/demo_register_new_tool.ipynb)



Moreover, if the tool is a `langchain tool`, you can directly use our `LangchainTool` to wrap and adapt with current frameworks.

```Python

from modelscope_agent.tools.langchain_proxy_tool import LangchainTool
from langchain.tools import ShellTool

# wrap langchain tools
shell_tool = LangchainTool(ShellTool())

print(shell_tool(commands=["echo 'Hello World!'", "ls"]))

```
