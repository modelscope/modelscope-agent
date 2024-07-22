# Callback

We have provided callback mechanism that allows users to customize code execution at key stages during Agent's execution, facilitating features like logging.

Currently, the customizable callback stages include the beginning and end of  `LLM, tool, rag` modules, as well as each call or round performed by `Agent`.

Users may inject their custom callback functions when instantiating an `Agent`. These functions are managed by the `CallbackManager` class and invoked at designated execution stages. Here is a simple example of defining and using a custom callback function:

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

Additionally, we offer the `RunStateCallback` for capturing intermediate states during the `Agent`'s operation, such as tool execution results or RAG recall results. When `RunStateCallback` is used, the results of each call can be accessed through the `run_states` attribute. An example structure of `run_states` could be as follows:
```Python
{
    # State per execution round
    1:
        [
            # Detailed execution results of modules such as llm/tool/rag
            RunState(type='llm', name='qwen-max', content='Action: RenewInstance\nAction Input: {"instance_id": "i-rj90a7e840y5cde", "period": "10"}\n', create_time=1720691946),
            RunState(type='tool_input', name='RenewInstance', content='{"instance_id": "i-rj90a7e840y5cde", "period": "10"}', create_time=1720691946),
            RunState(type='tool_output', name='RenewInstance', content="{'result': 'Renewal of ECS instance ID i-rj90a7e840y5cde completed, for a duration of 10 months.'}", create_time=1720691946)
        ],

    2:
        [
            RunState(type='llm', name='qwen-max', content='Renewal action completed for ECS instance ID i-rj90a7e840y5cde, for a duration of 10 months.', create_time=1720691949)
        ]
}
```
