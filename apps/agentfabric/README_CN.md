
<h1> Modelscope AgentFabric: 开放可定制的AI智能体构建框架</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

## 介绍

**Modelscope AgentFabric**是一个交互式智能体框架，用于方便地创建针对各种现实应用量身定制智能体。AgentFabric围绕可插拔和可定制的LLM构建，并增强了指令执行、额外知识检索和利用外部工具的能力。AgentFabric提供的交互界面包括：
- **⚡ 智能体构建器**：一个自动指令和工具提供者，通过与用户聊天来定制用户的智能体
- **⚡ 用户智能体**：一个为用户的实际应用定制的智能体，提供构建智能体或用户输入的指令、额外知识和工具
- **⚡ 配置设置工具**：支持用户定制用户智能体的配置，并实时预览用户智能体的性能

🔗 我们目前围绕DashScope提供的 [Qwen2.0 LLM API](https://help.aliyun.com/zh/dashscope/developer-reference/api-details) 来在AgentFabric上构建不同的智能体应用。同时我们正在积极探索，通过API或者ModelScope原生模型等方式，引入不同的举办强大基础能力的LLMs，来构建丰富多样的Agents。

## 安装

克隆仓库并安装依赖：

```bash
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent  && pip install -r requirements.txt && pip install -r demo/agentfabric/requirements.txt
```

## 前提条件

- Python 3.10
- 获取使用Qwen 2.0模型所需的API-key，可从[DashScope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)免费开通和获取。

## 使用方法

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/modelscope-agent
export DASHSCOPE_API_KEY=your_api_key
cd modelscope-agent/demo/agentfabric
python app.py
```

## 🚀 发展路线规划
- [x] 支持人工配置构建智能体
- [x] 基于LLM对话构建智能体
- [ ] 支持在ModelScope创空间上使用
- [ ] 知识库检索效果优化
- [ ] 支持智能体发布和分享
- [ ] 支持其他多种LLM模型API和ModelScope模型
- [ ] 处理长文本输入到内存
- [ ] 生产级支持：日志和性能分析
- [ ] 支持智能体微调
- [ ] 在不同场景中智能体的效果评估
