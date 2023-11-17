
<h1> Modelscope Agents：开源、可定制，重塑AI应用构建 </h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

## 介绍

**Modelscope Agents Builder**，一个交互式智能体框架，可以基于开源/商业LLM生成用户定制的智能体，用于实际应用，包含指令、额外知识和工具。交互界面包括：
- **智能体构建器**：一个自动指令和工具提供者，通过与用户聊天来定制用户的智能体
- **用户智能体**：一个为用户的实际应用定制的智能体，提供构建智能体或用户输入的指令、额外知识和工具
- **配置设置工具**：支持用户定制用户智能体的配置，并实时预览用户智能体的性能

## 安装

克隆仓库并安装依赖：

```bash
git clone https://github.com/modelscope/modelscope-agent.git
cd modelscope-agent  && pip install -r requirements.txt && pip install -r demo/agents/requirements.txt
```

## 前提条件

- Python 3.10
- Qwen系列API密钥，可以从[dashscope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.73d348f4zPlBdu)获取

## 使用方法

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/modelscope-agent
export DASHSCOPE_API_KEY=your_api_key
cd modelscope-agent/demo/agents
python app.py
```

## 路线图
- [x] 支持人工配置构建智能体
- [x] 基于llm对话构建智能体
- [ ] 支持在modelscope创空间上使用
- [ ] 知识库检索效果优化
- [ ] 支持智能体发布和分享
- [ ] 支持其他开源模型和商业API
- [ ] 处理长文本输入到内存
- [ ] 生产级支持：日志和性能分析
- [ ] 支持智能体微调
- [ ] 在不同场景中智能体的效果评估
