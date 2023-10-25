---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：multi-modal/
 - multi-modal 
tags: #自定义标签
- agent
- 文图生成

#datasets: #关联数据集
#  evaluation: 
#  #- damotest/beans
#  test:
#  #- damotest/squad
#  train:
#  #- modelscope/coco_2014_caption
models: #关联模型
- damo/ModelScope-Agent-7B
deployspec: #部署配置，默认上限CPU4核、内存8GB、无GPU、单实例，超过此规格请联系管理员配置才能生效
# 部署启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
  entry_file:  app_ms_agent_llm.py
# CPU核数
  cpu: 7
# 内存（单位MB)
  memory: 26500
# gpu个数
  gpu: 1 
# gpu共享显存（单位GB，当gpu=0时生效，当gpu>0时显存独占，此配置不生效）
  gpu_memory: 0
# 实例数
  instance: 1
  instance_type: ecs.gn7i-c8g1.2xlarge
license: Apache License 2.0
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/damo/Story-Agent.git
```
