## Tools

We provide some default pipeline tools of multiple domain that integrates in modelscope.

Also, you can custom your tools by inheriting base tool and define names, descriptions, and parameters according to pre-defined schema. And you can implement `_local_call()` or `_remote_call()` according to your requirement. An example of custom tool is provided below in [custom_tool](../../agent_scope/tools/custom_tool.py):

```python
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

```

You can also refer to our provided demo for a guide on how to use the new tool in [demo_register_new_tool](../../demo/demo_register_new_tool.ipynb)



Moreover, if the tool is a `langchain tool`, you can directly use our `LangchainTool` to wrap and adapt with current frameworks.

```Python

from agent_scope.tools import LangchainTool
from langchain.tools import ShellTool, ReadFileTool

# wrap langchain tools
shell_tool = LangchainTool(ShellTool())

print(shell_tool(commands=["echo 'Hello World!'", "ls"]))

```

## Output Wrapper

In certain scenarios, the tool may produce multi-modal data like images, audio, video, etc. However, this data cannot be directly processed by llm. To address this issue, we have implemented the `OutputWrapper` class. This class encapsulates the multi-modal data and returns a string representation that can be further processed by llm.

To use the `OutputWrapper` class, simply initialize an object with the origin multi-modal data, the `__init__()` function will save this data to a pre-defined local directory.

Then the `__repr__()` function of the OutputWrapper class then returns a string that concatenates the stored path and an identifier, and this can be used by llm for further processing.

```Python
class OutputWrapper:
    """
    Wrapper for output of tool execution when output is image, video, audio, etc.
    In this wrapper, __repr__() is implemented to return the str representation of the output for llm.
    Each wrapper have below attributes:
        path: the path where the output is stored
        raw_data: the raw data, e.g. image, video, audio, etc. In remote mode, it should be None
    """

    def __init__(self) -> None:
        self._repr = None
        self._path = None
        self._raw_data = None

        self.root_path = os.environ.get('OUTPUT_FILE_DIRECTORY', None)

    def __repr__(self) -> str:
        return self._repr

    @property
    def path(self):
        return self._path

    @property
    def raw_data(self):
        return self._raw_data
```
