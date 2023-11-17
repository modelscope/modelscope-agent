<h1> Modelscope Agents Builder: A new way to custom your agent with open source LLM </h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>


## Introduction

**Modelscope Agents Builder**, an interactive agents framework to generate user's customized agents for real-world applications, based on open-source/commercial LLMs with instructions, extra knowledge, and tools. The interactive interface including:
- **Builder agent**: an automatic instructions and tools provider for customizing user's agents by chatting with user
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
- Qwen max/plus Api key [from dashscope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.73d348f4zPlBdu)

## Usage

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/modelscope-agent
export DASHSCOPE_API_KEY=your_api_key
cd modelscope-agent/demo/agents
python app.py
```

## Roadmap

- [on going] Support chatglm3-6b
- [on going] handle long text input to memory
- [todo] support multi-users preview on modelscope space
- [todo] allow upload user customized agents to modelscope space
