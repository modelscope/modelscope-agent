---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：cv/nlp/audio/multi-modal/AutoML
- multi-modal
tags: #自定义标签
  - agent
  - AgentFabric
## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)

deployspec:
  entry_file: app.py

license: Apache License 2.0
---
<h1> Modelscope AgentFabric: Customizable AI-Agents For All</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

## Introduction
**ModelScope AgentFabric** is an interactive framework to facilitate creation of agents tailored to various real-world applications. AgentFabric is built around pluggable and customizable LLMs, and enhance capabilities of  instrcution following, extra knowledge retrieval and leveraging external tools. The AgentFabric is woven with interfaces including:
- ⚡ **Agent Builder**: an automatic instructions and tools provider for customizing user's agents through natural conversational interactions.
- ⚡ **User Agent**: a customized agent for building real-world applications, with instructions, extra-knowledge and tools provided by builder agent and/or user inputs.
- ⚡ **Configuration Tooling**: the interface to customize user agent configurations. Allows real-time preview of agent behavior as new confiugrations are updated.

🔗 We currently leverage AgentFabric to build various agents around [Qwen2.0 LLM API](https://help.aliyun.com/zh/dashscope/developer-reference/api-details) available via DashScope. We are also actively exploring
other options to incorporate (and compare) more LLMs via API, as well as via native ModelScope models.


## Installation
Simply clone the repo and install dependency.
```bash
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent  && pip install -r requirements.txt && pip install -r apps/agentfabric/requirements.txt
```

## Prerequisites

- Python 3.10
- Accessibility to LLM API service such as [DashScope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) (free to start).

## Usage

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/modelscope-agent
export DASHSCOPE_API_KEY=your_api_key
cd modelscope-agent/apps/agentfabric
python app.py
```

## 🚀 Roadmap
- [x] Allow customizable agent-building via configurations.
- [x] Agent-building through interactive conversations with LLMs.
- [x] Support multi-user preview on ModelScope space. [link](https://modelscope.cn/studios/wenmengzhou/AgentFabric/summary) [PR #98](https://github.com/modelscope/modelscope-agent/pull/98)
- [x] Optimize knowledge retrival. [PR #105](https://github.com/modelscope/modelscope-agent/pull/105) [PR #107](https://github.com/modelscope/modelscope-agent/pull/107) [PR #109](https://github.com/modelscope/modelscope-agent/pull/109)
- [x] Allow publication and sharing of agent. [PR #111](https://github.com/modelscope/modelscope-agent/pull/111)
- [ ] Support more pluggable LLMs via API or ModelScope interface.
- [ ] Improve long context via memory.
- [ ] Improve logging and profiling.
- [ ] Fine-tuning for specific agent.
- [ ] Evaluation for agents in different scenarios.
