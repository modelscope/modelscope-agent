# Data Science Assistant By ModelScope Agent

### 背景

[Modelscope-Agent](https://github.com/modelscope/modelscope-agent)是开源的agent框架，可通过vllm、ollama等工具接入各主流开源模型，也可以直接调用模型api；同时，提供RAG组件支持开发者快速接入知识库；最后丰富的工具生态，支持了大量的Modelscope社区模型作为工具，也支持了langchain的工具直接调用，除此之外还接入了各类常用的工具，如：web-browsing、文生图、code-interpreter等多个工具的能力。基于以上能力，Modelscope-Agent支持了开源GPTs能力——AgentFabric，可以让更多开发者交互式地构建个人agent助理，而不需要进行具体的代码开发。

综上，开发者也可以基于Modelscope-Agent框架自由定义符合自己需求的Agent，本文所述的Data Science Assistant就是基于Modelscope-Agent框架开发的解决数据科学问题的助手。

### 简介

Data Science Assistant（下文称DS Assistant）是基于modelscope-agent框架开发的数据科学助手，可以根据用户需求自动执行数据科学任务中探索性数据分析（EDA），数据预处理，特征工程，模型训练，模型评估等步骤完全自动化。

传统的ReACT框架对于简单的任务比较有效，但是具有以下几个主要的缺点：

1.  每次工具调用都需要一个 LLM 调用。

2.  LLM 一次仅计划 1 个子问题。这可能会导致任务的轨迹更加不可控，因为它不会被迫“推理”整个任务。


DS Assistant使用了plan-and-excute框架，这是一种新兴的Agent框架，旨在解决通过计划和执行步骤高效完成复杂任务。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqr7Zm6Aa8Oako/img/ed543b70-6690-4d19-b2b8-209963e781c8.png)
*langchain官网对Plan-and-execute Agent的描述 [https://blog.langchain.dev/planning-agents/](https://blog.langchain.dev/planning-agents/)*

以下是其工作流程：

1.  **任务计划**：代理接收用户输入的任务描述，进行语义理解，将任务分解为多个可执行子任务。

2.  **子任务调度**：基于任务之间的依赖关系和优先级，智能调度子任务的执行顺序。

3.  **任务执行**：每个子任务分配给特定的模块执行，

4.  **结果整合**：汇总各子任务的结果，形成最终输出，并反馈给用户。

### 快速上手
参考 [DataScience Assistant上手示例](https://github.com/modelscope/modelscope-agent/blob/master/examples/agents/data_science_assistant.ipynb)

### 架构方案

![/Users/pemppeng/Downloads/Architecture (4).png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/c64c41b0-5316-46e3-97be-b0477c2a0d5d.png)

整套系统包含4个主要模块：

1.  Plan模块：负责根据用户的需求生成一系列Task列表，并对task先后顺序进行拓扑排序，实现

2.  Execution 模块：负责任务的具体执行，保存任务执行结果。

3.  Memory management 模块：负责记录任务中间执行结果，代码，数据详情等信息。

4.  DS Assistant：作为整个系统的大脑，负责调度整个系统的运转。


## 执行过程

1.  任务规划阶段：


在这一阶段，DS Assistant根据用户输入的复杂数据科学问题，自动将其分解为多个子任务。这些子任务[根据依赖关系和优先级被组织和调度](https://arxiv.org/abs/2402.18679)，确保执行顺序符合逻辑且高效。

![/Users/pemppeng/Downloads/Plan (3).png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/11d9c0a1-4637-4e56-bf63-ad6abf1d96b8.png)

2.  执行阶段：每个子任务被具体化为可执行的操作，如数据预处理、模型训练等。


![/Users/pemppeng/Downloads/execution (3).png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/56f843f3-58a8-4427-996f-3aca6db8053b.png)

在所有Task执行完成后，DS Assistant 会将中间数据的执行情况 ( 包括每个task生成的代码和结果，消耗token数，任务时间 ) 保存为文件。

## 应用案例：

下面，我们以一个具体的例子来了解DS Assistant的执行过程：

我们选用Kaggle 上的一个比赛任务 **ICR - Identifying Age-Related Conditions** 作为示例：

该任务是一项机器学习任务，主要目的是通过分析各种数据（如医疗记录、基因数据、生活方式数据等），识别与年龄相关的健康状况。能够帮助医疗专业人员及早发现老年人群中常见的健康问题，并提供个性化的预防和治疗方案。

首先，我们需要配置好选用的LLM配置。我们引入MetaGPT的Data Science工具和Tool Recommender，可以根据任务类型向DS assistant推荐合适的数据科学工具。

接着，我们需要将任务的具体要求传给DS Assistant（注意，需要在要求中向DS Assistant指明数据文件的路径）
```python
from modelscope_agent.agents.data_science_assistant import DataScienceAssistant
from modelscope_agent.tools.metagpt_tools.tool_recommend import TypeMatchToolRecommender

llm_config = {
    'model': 'qwen2-72b-instruct',
    'model_server': 'dashscope',
}
tool_recommender = TypeMatchToolRecommender(tools=["<all>"])
ds_assistant = DataScienceAssistant(llm=llm_config, tool_recommender=tool_recommender)
ds_assistant.run(
    "This is a medical dataset with over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions. The target column is Class. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report F1 Score on the eval data. Train data path: ‘./dataset/07_icr-identify-age-related-conditions/split_train.csv', eval data path: ‘./dataset/07_icr-identify-age-related-conditions/split_eval.csv' ."
)
```
## Plan阶段

DS Assistant会根据用户需求生成任务列表，将整个数据处理流程进行分解，接着对任务列表进行按顺序处理。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/14a0d5ae-5183-4acb-8eb1-dbc3e61f5a9d.png)

可以看到，DS Assistant生成了5个任务，分别是数据探索，数据预处理，特征工程，模型训练和预测。

## Execute阶段：

### Task 1: 数据探索

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/d155ac02-22e8-465f-a601-919bf76c8d9c.png)

可以看到生成的代码在执行时报了如下错误，原因是没有引入numpy包。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/9ca04210-063a-457d-a176-b54b32134237.png)

DS assistant根据报错进行了反思，并重新生成代码并执行，成功输出数据探索的结果

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/c5cceb48-ba81-40a5-a598-764d40aa5b95.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/31886c6f-9cd2-4b9a-813e-28ef1363f7f0.png)

最后，code judge会对代码进行质检，确保本次生成代码逻辑正确。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/6d461003-a419-4f46-8be7-4d8cba7510b1.png)

### Task 2: 数据预处理

在数据预处理阶段，DS Assistant 分别对数值型数据和类别型数据进行了合适的缺失值处理，并清除了ID列。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/c85283ea-7378-4761-bfa1-7a3e40685b63.png)

### Task 3 特征工程

在修复了两次错误后，DS Assistant对数据进行了特征工程的处理，对类别型变量进行编码。

同时对之前定义的categorical\_columns变量进行了更新，去除了ID列。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/bc0135b8-16f2-4da8-8a34-0fd5af1fc679.png)

### Task 4 模型训练

DS Assistant主动安装了合适的依赖，并选择了多个模型（随机森林，梯度提升，逻辑回归）进行训练，并选择了结果最好的模型。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/88769a75-7cd1-48c9-b866-773c1bc90fa3.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/cda3014f-b18d-40f7-ae22-ac606131da9b.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/911e9c0a-3d40-4a52-b875-e3f3b0636fa2.png)

### Task 5 模型验证：

DS Assistant选择了训练集中F1分数最高的模型对验证集进行测试，并计算了这个模型在验证集上的F1分数，成功地完成了任务。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/f2aa8534-4ec3-4594-a6bc-cb0cca723b09.png)

### 结果保存

DS assistant支持将运行结果保存为Jupyter Notebook类型的文件，并记录运行的中间过程。

Jupyter Notebok

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/b88c57b4-8d3d-45a3-8357-223f50c64ef5.png)

中间过程记录JSON文件

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/52df52b1-8a3f-4b25-81c2-16e37336d31a.png)

# 实验效果

我们使用 [Data Interpreter: An LLM Agent For Data Science](https://arxiv.org/abs/2402.18679)的ML-Benchmark作为测试集，分别从_Normalized performance score (NPS)_，total time，total token三个维度对DS assistant效果进行评测。

其中NPS的计算公式如下：

Normalized performance score(规范化性能得分)是一种将不同任务或模型的性能指标标准化的方法，以便于比较。在不同的任务或模型评估中，可能使用不同的性能指标，如准确率(accuracy)、F1分数、AUC(曲线下面积)、RMSLE(均方根对数误差)等。由于这些指标的量纲和优化方向可能不同(有些越小越好，有些越大越好)，因此需要进行规范化处理。

规范化性能得分(NPS)的计算通常涉及以下步骤：

1.  确定指标优化方向：首先判断所使用的性能指标是"越大越好"还是"越小越好"。

2.  规范化计算：

    1.  如果指标是"越大越好"(例如准确率、F1分数、AUC)，则NPS等于原始值，因为这些指标不需要转换，较高的值已经表示较好的性能。

    2.  如果指标是"越小越好"((例如RMSLE)，则NPS通过一个函数转换，比如$1/(1+s)$，其中s是原始的性能值。这种转换确保了较小的损失值映射到接近1的较高NPS值。


规范化后的性能得分范围通常是0到1，其中1表示最优性能，而0表示最差性能。

实验任务详情和结果如下( 绿色代表当前任务下最优指标 )

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/87ac9117-2a9c-478e-988e-2993c47d0017.png)
*ML-Benchmark 任务详情*

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlKaVzEkV0qDvL/img/d751324e-9ac5-4751-a1dd-8f10b9835514.png)
*DS Assistant 和 开源SOTA 在ML-Benchmark 效果对比（其中开源SOTA效果指MetaGPT 实测值）*

结果说明，在部分复杂的数据科学任务上，DS Assistant在规范化性能得分(NPS)，任务时间，消耗token数的指标上取得超过开源SOTA的效果。

实验日志已保存至 https://modelscope-agent.oss-cn-hangzhou.aliyuncs.com/resources/DS_Assistant_results.zip


## 结语：

综上，利用Data Science Assistant，

1. 不熟悉数据分析流程但是有需要去对数据进行分析的同学可以快速地根据生成的任务以及处理过程，了解处理数据的思路，以及技术点

2. 对于了解数据分析流程的同学，可以通过详细的描述，来影响数据处理的方法，方便做不同的实验参照比较。

3. 对于所有人，可以自动化的快速实现对于当前手上文件的更深层次的理解，仅需提问即可。


后续工作

1.  进一步提高任务执行成功率：

    1.  对于Code Agent来说，传入信息量过大（报错信息，中间数据信息，已生成代码信息）会导致模型生成代码正确率下降，可以在未来考虑使用LLM进行总结，对信息进行筛选。

    2.  同一个Task可进行进一步的分解，以降低对LLM推理能力的要求。

2.  对话交互式，可以将任务和任务的执行展示分开，通过对话的方式推进任务，并影响执行结果

3.  支持批处理相同任务多批文件的场景。
