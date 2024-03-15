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

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<a href="https://github.com/modelscope/modelscope"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.3-5D91D4.svg"></a>
<a href="https://github.com/modelscope/modelscope-agent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/modelscope-agent"></a>
<a href="https://github.com/modelscope/modelscope-agent/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
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

Another demo shows it could work on multi-modality video generation task. (*polishing demo code, will add soon*).

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/video-generation-multi-agent.png" width="600" />
</p>

## Future works

Even though, we have designed such multi-agent system, it still has many challenges to get into production environment.
The following problem are known issues:
* In a single-machine-multi-processes task, *Ray* is out-powered in such scenario, and has some overhead on initialization and multi-processes communication.
* *Ray* still have many issues on fork process, which might cause problems running with *Gradio*.
* User has to write code for different tasks, even for the rule based task, there is no silver bullet task template to avoid coding for task step forward.
* Hard to debugging for those who only code on single process, it should be better to track issues by logs.

Other than above issues, following features still need to be added, in order to make multi-agent fit in more complex task
* Support more complicated distributed task, such as the data spider system mentioned above
* Better instruction & prompt for different task
* Distributed memory
* Distributed tool service
* TBD
