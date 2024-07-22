# Callback模块使用说明

我们设计了一套回调机制，允许用户可以自定义的在`Agent`执行的关键节点中插入想要执行的代码，便于实现诸如日志记录等功能。

目前，我们提供的可插入回调函数的节点位置包括`Agent`中常见的`LLM,tool,rag `等模块的执行以及`Agent`本身每次/轮执行的开始和结束。

用户可以在Agent实例化的时候传入想要使用的回调函数。这些回调函数会通过`CallbackManager`类统一管理，并在相应的节点执行。一个简单的自定义回调函数的定义和使用样例如下所示：

```Python
class SimplaCallback(BaseCallback):
    def on_llm_start(self, *args, **kwargs):
        print('start calling llm')

    def on_rag_start(self, *args, **kwargs):
        print('start calling rag')

    def on_tool_start(self, *args, **kwargs):
        print('start calling tool')

...
bot = RolePlay(function_list=function_list,llm=llm_config, instruction=role_template, callbacks=[callback], stream=False)

bot.run('xxx')
```

## RunStateCallback

我们提供了`RunStateCallback`，用于记录Agent执行过程中的中间状态，例如工具的调用/执行结果，RAG的召回结果等。如果使用了`RunStateCallback`，可以通过`run_states`属性获取中间调用的结果。一个可能的`run_states`格式如下：

```Python
{
    # 每轮的run_state
    1:
        [
            # llm/tool/rag调用的具体结果
            RunState(type='llm', name='qwen-max', content='Action: RenewInstance\nAction Input: {"instance_id": "i-rj90a7e840y5cde", "period": "10"}\n', create_time=1720691946),
            RunState(type='tool_input', name='RenewInstance', content='{"instance_id": "i-rj90a7e840y5cde", "period": "10"}', create_time=1720691946),
            RunState(type='tool_output', name='RenewInstance', content="{'result': '已完成ECS实例ID为i-rj90a7e840y5cde的续费，续费时长10月'}", create_time=1720691946)
        ]

    2:
        [
            RunState(type='llm', name='qwen-max', content='已经完成了ECS实例ID为i-rj90a7e840y5cde的续费操作，续费时长为10个月。', create_time=1720691949)
        ]
}
```
