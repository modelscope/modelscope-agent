<h1> Modelscope Agents: Open Source, Customizable, Reshaping AI Application Development </h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>


## Introduction

**Modelscope Agents Builder**, an interactive agents framework to generate user's customized agents for real-world applications, based on open-source/commercial LLMs with instructions, extra knowledge, and tools. The interactive interface including:
- **Agent Builder**: an automatic instructions and tools provider for customizing user's agents by chatting with user
- **User agent**: a customized agent for user's real-world applications, with instructions, extra-knowledge and tools provided by builder agent or user inputs
- **Configuration setting tool**: support user to customize the configuration of user agent, and preview the performance of user agent in real-time

## Installation

clone repo and install dependency：

```bash
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent  && pip install -r requirements.txt && pip install -r demo/agents/requirements.txt
```

## Prerequisites

- Python 3.10
- Qwen-Series API key which can be obtained from [dashscope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.73d348f4zPlBdu)

## Usage

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/modelscope-agent
export DASHSCOPE_API_KEY=your_api_key
cd modelscope-agent/demo/agents
python app.py
```

## Roadmap
- [x] Support using configuration to build agent
- [x] Implement agent builder to build agent through conversation with llm
- [ ] Support multi-users preview on modelscope space
- [ ] Optimize knowledge retrival performance
- [ ] Support pusblishig agent and sharing agent
- [ ] Support other opensource models and commercial api.
- [ ] Handle long text input to memory
- [ ] Production-level support: logging and profiling
- [ ] Finetuning for specific agent
- [ ] Evaluation for agents in different scenario
