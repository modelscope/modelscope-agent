<h1 align="center"> Multi-Agent based on ModelScope-Agent and Ray</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a> ｜ <a href="https://arxiv.org/abs/2309.00986">Paper</a> ｜ <a href="https://modelscope.cn/studios/damo/ModelScopeGPT/summary">Demo</a>
<br>
        <a href="README.md">English</a>&nbsp ｜ &nbsp中文
</p>



## 介绍

大型语言模型（LLM）智能体的应用已经变得非常普遍。
然而，单智能体系统(single-agent)在复杂的互动场景中，例如斯坦福小镇、软件公司、多方辩论等，经常会遇到困难。
因此，多智能体（multi-agent）架构被提出来解决这些限制，并且现在也已经有广泛的使用。
为了让ModelScope-Agent能够运行在multi-agent模式下，我们提出了以下multi-agent架构。


## 架构

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/multi-agent_with_modelscope.png" width="600" />
</p>

## 动机

在我们的设计中，[Ray](https://docs.ray.io/en/latest/)在扮演着重要的角色。
通过Ray，我们可以通过只更新当前项目中的几行代码，就能轻松地将ModelScope-Agent扩展到分布式multi-agent系统，
并让我们的应用程序准备好进行并行处理，而无需关心服务通信、故障恢复、服务发现和资源调度。

为什么multi-agent框架需要这么复杂的能力呢？

当前的multi-agent框架主要关注于使用不同的single-agent来完成任务以获得更好的结果，许多论文已经证明multi-agent比single-agent获得了更好的结果。
然而，在现实中，许多任务应该由一群single-agent高效完成，例如一个数据爬虫任务可能需要数百个爬虫agent和数十个数据处理agent。
目前，在这个场景下的的multi-agent框架还很少，另外能够支持chatbot场景的multi-agent框架基本没有。
另一方面，ModelScope-Agent已经证明可以在生产环境中工作[ModelScope Studio](https://modelscope.cn/studios/agent)，
因此，我们相信将single-agent扩展到分布式multi-agent可能是multi-agent在生产环境中落地的一个趋势。


## 方法
考虑到ModelScope-Agent的当前状态，希望不影响现有的工作，我们提出了以下设计解决方案：


1. **将multi-agent的交互逻辑与single-agent的逻辑解耦:**
   - 使用**[AgentEnvMixin](../agent_env_util.py)**类基于Ray处理所有multi-agent通信逻辑，无需更改任何现有single-agent模块中的原始逻辑。
   - 在**[Environment](../environment.py)**模块中管理环境信息，使用发布/订阅机制来推动agent之间的互动，而不会在执行层面阻塞agent。
   - 消息中心维护在Environment模块中，同时每个各个agent也单独管理自己的历史记录。

2. **引入agent注册中心[Agent Registry Center](../agents_registry.py)概念:**
   - 用于维护系统中agent的信息并实现相关能力扩展。
   - 用于更新agent状态。

3. **引入任务中心[Task Center](../task_center.py)概念:**
   - 设计的任务中心具有开放性，允许将消息订阅或发布给所有agent，支持agent之间各种形式的交互，如随机交流，或者通过用户定义的逻辑以循环方式进行推进。
   - 允许通过使用`send_to` 和`sent_from` 的方法直接交互方法，可快速开发流程简单的应用。
   - 支持聊天机器人模式和终端模式，使用户可以在流媒体聊天gradio应用程序或终端上运行multi-agent。


## 示例

基于有god角色的的多人聊天室gradio [app](../../demo/demo_multi_roles_chat_room.ipynb)

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/multi-roles-chat-room.png" width="600" />
</p>

基于固定流程的视频生成gradio [app](../../demo/demo_multi_role_videogen.ipynb).

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/video-generation-multi-agent.png" width="600" />
</p>

## 快速开始

本设计中Multi-Agent System主要以面向流程设计的方式在Ray上运行。
这样，用户不需要关心任何额外的分布式或多进程问题，ModelScope-Agent和Ray已经涵盖了这部分。
用户只需要根据任务类型编写过程脚本，以驱动代理之间的通信。

运行multi-agent分为两个阶段，初始化和处理，如下图所示。
<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/sequence_diagram.png" width="600" />
</p>

在初始化阶段，Ray会通过同步操作将所有类转换为actor，例如：`task_center`、`environment`、`agent_registry`和`agent`。

### 任务中心

任务中心(`task_center`)将使用 `environment` 和 `agent_registry` 两个组件来推进任务，并管理任务过程。
其中 `remote=True` 被用来允许*Ray*在此过程中扮演核心角色,用户不需要关心分布式或多进程的细节。
在运行任务之前，如果我们想在多进程中运行任务，必须先初始化*Ray*， 并将`task_center`转换为*Ray*上的 actor。
在`task_center`中，`environment`和`agent_registry`也会被自动转换为*Ray*上的 actor。

以下代码是`task_center`的初始化。请注意，使用`ray.get()`确保初始化操作是同步的。

```python3
import ray
from modelscope_agent import create_component
from modelscope_agent.task_center import TaskCenter
from modelscope_agent.multi_agents_tasks.executors.ray import RayTaskExecutor

REMOTE_MODE = True

if REMOTE_MODE:
    RayTaskExecutor.init_ray()

task_center = create_component(
    TaskCenter,
    name='task_center',
    remote=REMOTE_MODE)
```

### Agents
所有的agent将通过函数`create_component`进行初始化。
如果设置`remote=True`，将把agent转换为一个*Ray*上的`actor`，并将在一个独立的进程上运行.
如果`remote=False`，那么它就是一个简单的agent类，和普通的single agent没有区别。


入参`name`用于定义在*Ray*中对应的`actor`的名称；
另一方面，入参`role`用于在ModelScope-Agent中定义角色名称。

其余输入的定义与 [single agent](../agent.py)相同。

```python3
import os

from modelscope_agent import create_component
from modelscope_agent.agents import RolePlay

llm_config = {
    'model': 'qwen-max',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
    'model_server': 'dashscope'
}
function_list = []
role_play1 = create_component(
    RolePlay,
    name='role_play1',
    remote=True,
    role='role_play1',
    llm=llm_config,
    function_list=function_list)

role_play2 = create_component(
    RolePlay,
    name='role_play2',
    remote=True,
    role='role_play2',
    llm=llm_config,
    function_list=function_list)
```

这些agent随后将通过task_center的`add_agents`方法进行注册。

```python
# register agents in remote = True mode
ray.get(task_center.add_agents.remote([role_play1, role_play2]))
```

如果先要不用ray，只用`remote=False`，则不需要使用`ray.get()`，我们可以把这段代码换成如下：
```python
# register agents in remote = False mode
task_center.add_agents([role_play1, role_play2]))
```
值得注意的是，目前为主以上所有操作都是以同步方式进行的，以确保所有的actor都正确初始化。 不管remote mode状态如何

### 任务处理

我们可以通过调用`send_task_request`来启动一个新的任务，并将任务发送到`environment`。
```python
task = 'what is the best solution to land on moon?'
ray.get(task_center.send_task_request.remote(task))
```
另外，我们也可以通过参数`send_to`传入agent的角色名称，以向特定的agent发送任务请求。
```python
ray.get(task_center.send_task_request.remote(task, send_to=['role_play1']))
```

对应的`remote=False`的情况，我们可以直接调用`send_task_request`方法，而不需要使用`ray.get()`。
```python
task_center.send_task_request(task)
```
以及
```python
task_center.send_task_request(task, send_to=['role_play1'])
```

然后，我们可以使用task_center的静态方法`step`来编写我们的multi-agent流程逻辑。
```python
import ray
n_round = 10
while n_round > 0:

    for frame in task_center.step.remote():
        print(ray.get(frame))


    n_round -= 1
```

`step`方法需要被转换为*Ray*中的task函数，即`step.remote`，因此我们必须将其设置为静态方法，并将task_center作为输入传入，以便让这个step函数获得任务的信息。
在`step`任务方法内部，它将并行地调用对于那些在这一步骤中应该响应的agent的`step`方法。

返回的响应将是一个分布式生成器，在*Ray*中被称为`object reference generator`，它是*Ray*集群中的一个共享内存对象。
因此，我们必须调用`ray.get(frame)`来将这个对象提取为正常的生成器。

要详细了解ray，请参考Ray介绍[文档](https://docs.ray.io/en/latest/ray-core/key-concepts.html)。


在 `remote=False` 的情况下，我们可以直接调用`step`方法，而不需要使用`ray.get()`。
```python
n_round = 10
while n_round > 0:

    for frame in task_center.step():
        print(frame)

    n_round -= 1
```
这里返回的是一个标准的python生成器，我们可以直接使用。


### 总结

到目前为止，我们创建了一个包含两个agent的multi-agent system，并让它们讨论一个关于 *登月的最佳解决方案是什么* 的话题。

同时，值得注意的是，随着agent数量的增加，这个基于*Ray*的multi-agent system的效率将会显现出来, 数量增大能发挥出来*Ray*的特性。

目前我们实现了一个非常简单的任务，我们希望开发者能够探索更多具有更复杂条件的任务，我们也会持续的进行探索。


## 未来工作

尽管我们设计了这样的multi-agent框架，但要将其投入生产环境还有许多挑战。以下问题是已知问题：

* 在单机多进程任务中，Ray在这种场景中不占优势，并且在初始化和多进程通信上有一些额外开销，影响速度。
* *Ray*在fork进程方面还有许多问题，例如使用*Gradio* 启动任务的时候会有ray的子进程退出问题。
* 用户必须为不同的任务编写代码，即使对于基于规则的逻辑，也没有万能的解决方案来避免为写代码。
* 对于那些只在单进程上编码的人来说，很难调试，通过日志追踪问题应该会更好。

除了上述问题之外，为了使multi-agent适应更复杂的任务，还需要添加更多功能，包括：
* 适配如上文中提到的数据爬虫任务的更复杂的任务场景。
* 为不同任务提供更好的指令和提示。
* 分布式memory管理。
* 分布式tool服务调用。
* TBD
