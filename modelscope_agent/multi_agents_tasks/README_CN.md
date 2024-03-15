<h1> ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models</h1>

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

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<a href="https://github.com/modelscope/modelscope"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.3-5D91D4.svg"></a>
<a href="https://github.com/modelscope/modelscope-agent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/modelscope-agent"></a>
<a href="https://github.com/modelscope/modelscope-agent/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
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

基于固定流程的视频生成gradio app (*polishing demo code, will add soon*).

<p align="center">
  <img src="https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/video-generation-multi-agent.png" width="600" />
</p>

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
