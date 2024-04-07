<h1 align="center"> Multi-Agent based on ModelScope-Agent and Ray</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a> ｜ <a href="https://arxiv.org/abs/2309.00986">Paper</a> ｜ <a href="https://modelscope.cn/studios/damo/ModelScopeGPT/summary">Demo</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbsp 英文
</p>



## Introduction

The application of LLM (Large Language Models) agents has become widespread.
However, single-agent systems often fall in complex interactive scenarios, such as Stanfordville, software companies, multi-party debates, etc.
Thus, Multi-agent architectures have been proposed to address these limitations and are now widely used.
In order to allow modelscope-agent run on multi-agent mode, we have proposed the following architecture.

## Architecture

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/multi-agent_with_modelscope.png" width="600" />
</p>

## Why

In our design, [Ray](https://docs.ray.io/en/latest/) is running an important role in multi-agent mode.
With Ray we could easily scale modelscope-agent into a distribution system by only updating couples of lines code in current project,
and make our application ready for parallel processing without taking care about Service Communication, Fault Recovery, Service Discovery and Resource Scheduling.

Why a multi-agent need such complicated abilities?

Current multi-agent is focusing on using different agents working on a task to get a better result, many papers have demonstrated that the multi-agent
get a better result than single agent. However, in reality, many tasks should be accomplished by a swarm of agents with efficiency,
such as hundreds of spider agents with dozens of data process agents for a data spider task.
No open-source multi-agent has design for this scenario with supporting chatbot mode.

On the other hand, modelscope-agent has been demonstrated that can work on a production environment [modelscope studio](https://modelscope.cn/studios/agent),
and we believe that scaling up single-agent into distributed multi-agent could be a trending for multi-agent working on production environment

## Method
Considering the current status of ModelScope-Agent, the following design solutions are proposed:

1. **Decouple multi-agent interactive logic from single-agent Logic:**
   - Use **[AgentEnvMixin](../agent_env_util.py)** class to handle all of multi-agent communication logic based on **Ray**, without changing any origin logic in single agent modules.
   - Extract environment information in **[Environment](../environment.py)** module, using a publishing/subscribe mechanism to advance interactions without execution-level blocking between agents.
   - Message hub is maintained in **Environment** module, meanwhile each multi-agent actor manage their own history

2. **Introduce an *[Agent Registry Center](../agents_registry.py)* Concept:**
   - Maintain system agents' information for capability expansion.
   - Update agent status

3. **Introduce a *[Task Center](../task_center.py)* Concept:**
    - Design the Task Center to be open-ended, allowing messages to be subscribed or published to all agents, sent to random agents, or delivered in a round-robin fashion by user defined logic.
    - Allow rapid development by using direct interaction methods like `send_to` and `sent_from`.
    - Support both of *Chatbot Mode* and *Terminal Mode*, such that user could run multi-agent in a streaming chat gradio apps or on a terminal

## Examples

A simple multi-roles chat room gradio [app](../../demo/demo_multi_roles_chat_room.ipynb) demonstrated the logic of multi-agent design.

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/multi-roles-chat-room.png" width="600" />
</p>

Another demo shows it could work on multi-modality video generation [task](../../demo/demo_multi_role_videogen.ipynb).

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/video-generation-multi-agent.png" width="600" />
</p>

## Quick Start

The multi-agent is running on Ray in a Process Oriented Design(POD) manner.
Such that, user don't need to take care any additional distribution or multi-processes stuff, modelscope-agent with ray have
covered this part. User only need to write the procedure based on the task type to drive the communication between agents.

There are two stages to run a multi-agent task, initialization and Process as shown below.


<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/sequence_diagram.png" width="600" />
</p>

During initialization stage, Ray will covert all classes into actor by a sync operation,
such as: `task_center`, `environment`, `agent_registry` and `agent`.

### Task Center
Task center will use `environment` and `agent_registry` to step forward the task, and manage the task process.
The `remote` is used to allow *Ray* to run a core role in this process.
The user don't need to care about the distribution or multi-processes stuff.
Before running the task in multi processes mode, we have to initialize the *Ray*, and convert the `task_center` into a ray actor
inside the `task_center`, the `environment` and `agent_registry` will be converted into actor on ray automatically.

The following code is the initialization of the task center.
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
Agents will be initialized by the function `create_component`.
The `remote=True` will convert the agent into a ray actor and will run on an independent process,
if `remote=False` then it is a simple agent.

`name` is used to define the agent actor's name in *Ray*, on the other hand,
`role` is used to define the role name in *ModelScope-Agent*.

The rest definition of the inputs are the same as [single agent](../agent.py).


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

Those agents will then be registered to `task_center` by `add_agents` method.
```python

# register agents in remote = True mode
ray.get(task_center.add_agents.remote([role_play1, role_play2]))
```
If you want to run the multi-agent in a single process without *Ray*, you could set `remote=False` in the agent initialization.
We have to slightly modify the `add_agents` method to support the single process mode.

```python

# register agents in remote = False mode
task_center.add_agents([role_play1, role_play2])
```

All the operations so far are in a sync manner, in order to make sure all the actors are correctly initialized.
No Matter in *Ray* mode or not.


### Task Process
We could start a new task by calling `send_task_request`, and send the task to the `environment`.

```python
task = 'what is the best solution to land on moon?'
ray.get(task_center.send_task_request.remote(task))
```
also we could send task request only to specific agents by passing the input `send_to` with role name of agent.
```python
ray.get(task_center.send_task_request.remote(task, send_to=['role_play1']))
```

The `remote=False` mode would be like this:
```python
task_center.send_task_request(task)
```
and
```python
task_center.send_task_request(task, send_to=['role_play1'])
```

Then, we could code our multi-agent procedure logic with task_center's static method `step`

```python
import ray
n_round = 10
while n_round > 0:

    for frame in task_center.step.remote():
        print(ray.get(frame))


    n_round -= 1
```
The `step` method should be converted to a `task` function in ray as `step.remote`, so we have to make it static,
and pass in the `task_center` as input, in order to let this step function have the information about this task.

Inside the `step` task method, it will call each agent's `step` method parallely, for those agents who should response in this step.
The response will be a distributed generator so-called `object reference generator` in ray, which is a memory shared object among the ray cluster.
So we have to call `ray.get(frame)` to extract this object as normal generator.

For detail understanding of ray, please refer the Ray introduction [document](https://docs.ray.io/en/latest/ray-core/key-concepts.html)


The `remote=False` mode will be much easier:
```python
n_round = 10
while n_round > 0:

    for frame in task_center.step():
        print(frame)

    n_round -= 1
```


### Summary

So far, we have built a multi-agent system with two agents, and  let then discuss a topic about
*'what is the best solution to land on moon?'*.

With the increasing number of agents, the efficiency of this multi-agent on ray will be revealed.

This is a very simple task, and we hope developers could explore more tasks with more complicated conditions, so as we will do.

And for the local mode without ray, only should we remove all of the `ray.get()`, `.remote()` and `ray` in the code.

## Future works

Even though, we have designed such multi-agent system, it still has many challenges to get into production environment.
The following problem are known issues:
* In a single-machine-multi-processes task, *Ray* is out-powered in such scenario, and has some overhead on initialization and multi-processes communication.
* *Ray* still have many issues on fork process, which might cause problems running with *Gradio*.
* User has to write code for different tasks, even for the rule based task, there is no silver bullet task template to avoid coding for task step forward.
* Hard to debugging for those who only code on single process, it should be better to track issues by logs.

Other than above issues, following features still need to be added, in order to make multi-agent fit in more complex task
* Support more complicated distributed task, such as the data spider system on multi-machine cluster
* Actually, we never try to run the multi-agents on a single process(`remote=True`) with python native async function, might be tested and added for some simple case.
* Better instruction & prompt for different task
* Distributed memory
* Distributed tool service
* TBD
