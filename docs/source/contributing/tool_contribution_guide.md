# 如何贡献tool

欢迎开发者贡献tool，可提交PR，参考：[PR示例](https://github.com/modelscope/modelscope-agent/pull/283/commits)

## 详细攻略：

### tool注册

历经多次迭代，用户可以更快速的接入一个新的工具进入主库。

在`modelscope_agent/tools`路径下创建您的工具类文件，这里假设创建名为`test_sambert_tool.py`的文件

以下为接一个新`tool`入库的示例：
```python
import os
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.tools.utils.output_wrapper import AudioWrapper

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')
@register_tool('test_sambert_tool')
class TestSambertTool(BaseTool):
    description = 'Sambert语音合成服务，将文本转成语音'
    name = 'test_sambert_tool'
    parameters: list = [{
        'name': 'text',
        'description': '需要转成语音的文本',
        'required': True,
        'type': 'string'
    }]
    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        self.api_key = self.cfg.get('dashscope_api_key',
                                    os.environ.get('DASHSCOPE_API_KEY'))
        if self.api_key is None:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')
        super().__init__(cfg)
    def call(self, params: str, **kwargs) -> str:
        from dashscope.audio.tts import SpeechSynthesizer
        params = self._verify_args(params)
        tts_text = params['text']
        if tts_text is None or len(tts_text) == 0 or tts_text == '':
            raise ValueError('tts input text is valid')
        os.makedirs(WORK_DIR, exist_ok=True)
        wav_file = WORK_DIR + '/sambert_tts_audio.wav'
        response = SpeechSynthesizer.call(
            model='sambert-zhijia-v1', format='wav', text=tts_text)
        if response.get_audio_data() is not None:
            with open(wav_file, 'wb') as f:
                f.write(response.get_audio_data())
        else:
            raise ValueError(
                f'call sambert tts failed, request id: {response.get_response().request_id}'
            )
        return str(AudioWrapper(wav_file))
```

解读：

- line6：`@register_tool('test_sambert_tool')`用于将该工具类注册到注册中心以便后续调用， 并取名为 test_sambert_tool。
- line 8-15：工具在被大模型调用的时候，所有需要的信息都会被定义在这部分，不同的工具对应的`name`, `description`必须要描述清晰，以便大模型能够正确的使用该工具。 于此同时， `parameters`需要严格按照上述格式进行定义，以便模型能够正确生成调用该工具的参数，参数需要包括：`name`, `description`, `required`和`type`。
- line 16: `__init__()`方法可以把一些非运行时相关的配置加在这里。
- line 17:  `self.cfg = cfg.get(self.name, {})`对于一些配置较多，或者需要配置来自于文件的场景，可以使用该方法来初始化一个工具。
- line 23: `call()`方法定义了该工具使用参数进行任务执行的具体方法，注意入参即为上一步中大模型生成的`parameters`，并且以`string`的方式传入。
- line 25:  `params = self._verify_args(params)`该方法用`parse`，`stringify`的 `parameters`成为`dict`，我们已经有默认方法得以实现。 针对一些模型生成效果不好的场景，或者需要有特殊解析逻辑的场景，用户可以自行实现该类。
- line 26-40: 该地方实现了利用`dashscope`的`tts`接口调用的方法，用户可以在这里完成实现各自`tool`的具体功能，对于调用一些异步的api，需要轮训等任务的，可以参考[这里](https://github.com/modelscope/modelscope-agent/blob/master/modelscope_agent/tools/dashscope_tools/style_repaint.py)

### 其他配置项
- 需要在`modelscope_agent/tools/base.py`中对register_map进行配置，添加`'test_sambert_tool': 'TestSambertTool'`到字典里，格式为`注册名：类名`。
- 需要在`modelscope_agent/tools/__init__.py`中对_import_structure进行配置，添加`'test_sambert_tool': ['TestSambertTool']`，格式为`文件名：类名`。

### 在Agent中使用tool
在上一步中，我们已经定义了一个新的`tool`类，那么下面会演示如何调用它。
```python
import os
from modelscope_agent.agents import RolePlay
role_template = '你扮演一名语音合成大师，可以将文本转化为语音。'
llm_config = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    }
# 对于需要额外config的情况
function_list = [{
    'test_sambert_tool':{
      'dashscope_api_key': os.environ.get('DASHSCOPE_API_KEY')
    }
}]
# 对于不需要额外config的情况，直接写注册的名称即可
function_list = ['test_sambert_tool']
bot = RolePlay(function_list=function_list,llm=llm_config, instruction=role_template)
response = bot.run("请帮我把，modelscope-agent真厉害，用甜美女声念出来。", remote=False, print_info=True)
text = ''
for chunk in response:
    text += chunk
print(text)
```

解读：

- line 2：定义了`agent`的提示词，用于推进任务。
- line 3-6: 定义了`agent`所需要的模型，目前`qwen-max`是qwen系列中指令理解，指令生成效果最好的模型。
- line 8-14: 在这里我们将刚才注册到工具注册中心的新工具`test_sambert_tool`加到`function_list`待`agent`调用，其中两种形式，带`config`传入的或者单纯名字的均可。
- line 15: 利用上述几步的信息初始化`agent`。
- line 16-20: 提交任务给`agent`，完成相关工具的调用。

另外，还有一个演示demo可以参考：请点击[这里](https://github.com/modelscope/modelscope-agent/blob/master/demo/demo_register_new_tool.ipynb)。

### 文档准备

- 需要添加`readme`，用于解释该`readme`中所调用的函数参数意义，或者是否需要额外的配置，以便能够跑通；
- 同时，还需要提交贡献者信息以便，系统能够透出展示贡献者信息；
- 相关readme示例如下：
```
# 语音合成
API详情地址：[https://help.aliyun.com/zh/dashscope/developer-reference/quick-start-13?spm=a2c4g.11186623.0.i4](https://help.aliyun.com/zh/dashscope/developer-reference/quick-start-13?spm=a2c4g.11186623.0.i4)
|  必选参数  |  参数解释  |
| --- | --- |
|  API密钥  |  这里复用DASHSCOPE\_API\_KEY，无需额外添加环境变量。|
|  贡献者昵称  |  XXX  |
| --- | --- |
|  邮箱  |  xxx@xxx |
|  魔搭账号  |  xxx  |
|  GitHub  |  [https://github.com/xxx](https://github.com/xxx)  |
```

或者参考其他tool的介绍文案：

![图片](../../resource/tool-readme.png)

### 添加单元测试
- 除了完成核心模块的开发，还需要添加单元测试用例，以便确保功能完整性，测试用例可以参考：[code interpreter unit test](../../tests/tools/test_code_interpreter.py)
- 跑测试用例，仅需要确保当前测试用例通过即可
- `modelscope-agent`利用`pytest`完成测试，因此未安装`pytest`的话，需要先安装`pytest`，具体跑测试的示例如下：

```shell
pytest modelscope-agent/tools/contrib/demo/test_case.py
```

### 总结：文档结构
综上，为了保证文档结构的规范，需要您对需要提交的代码进行文档结构的调整。

- 提交代码需要三个文件： 执行文件，测试文件和`readme`
- 需要将上述三个文件放到一个文件夹中，为了让该文件夹被引用还需要一个`__init__.py`文件
- 需要将测试时对`modelscope_agent/tools/__init__.py`中关于_import_structure的修改重新进行配置，参考如下示例。
- 最后，需要确保该文件夹位于 `modelscope_agent/tools/contrib` 下。

一个实例如下图所示：

```
contrib
├── __init__.py
└── demo
    ├── README.md
    ├── __init__.py
    ├── renew_aliyun_instance.py
    └── test_case.py
```
- 需要将该类加入到 `modelscope_agent/tools/contrib/__init__.py`， 以便被上一层引用到，具体参考[tools/contrib](../../../modelscope_agent/tools/contrib/demo)。
- 重新执行确保代码可行

### 代码提交规范

- 所有新加入的tool类，需要从 modelscope-agent 库中fork出来独立的库，并单独开分支提交， 命名规则为：
```shell
git checkout -b tool/{your-new-tool-name}
```
- 新增完代码以及对应的单测用例以外，需要确保代码格式正确，可以参考如下命令执行，确保安装自动格式检查\更新的工具pre-commit：

```shell
# 对于第一次跑的同学，需要安装pre-commit，并在项目中初始化
pip install pre-commit
pre-commit install
pre-commit run --all-files
# 后续在每一次commit的时候会有格式自动检查及修复, 使用如下命令即可
git add .
git commit -m "add new tool"
# 如果遇到无法进行修复的格式错误，可以尝试手动修改，或者跳过
git commit -m "add new tool" --no-verify
```

- 最后提交到代码库后，可以在[这里](https://github.com/modelscope/modelscope-agent/compare)通过提交pr的方式合并会主库，modelscope-agent主库维护者会尽快进行校验合并工作。
