---
license: Apache License 2.0
domain: #领域：cv/nlp/audio/multi-modal/AutoML
# - cv
tags: #自定义标签
- 
datasets: #关联数据集
  evaluation: 
  #- damotest/beans
  test:
  #- damotest/squad
  train:
  #- modelscope/coco_2014_caption
models:
- damo/cv_ddsar_face-detection_iclr23-damofd
- damo/cv_resnet101_image-multiple-human-parsing
- damo/cv_unet_skin-retouching
- damo/cv_resnet34_face-attribute-recognition_fairface
- damo/cv_manual_face-quality-assessment_fqa
- damo/cv_unet-image-face-fusion_damo
- damo/cv_ir_face-recognition-ood_rts
- damo/cv_manual_facial-landmark-confidence_flcm
- Cherrytest/rot_bgr
- ly261666/cv_portrait_model
- ly261666/civitai_xiapei_lora
deployspec: #部署配置，默认上限CPU4核、内存8GB、无GPU、单实例，超过此规格请联系管理员配置才能生效
# 部署启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# entry_file: 
# CPU核数
  cpu: 7
# 内存（单位MB)
  memory: 26500
# gpu个数
  gpu: 1
# 实例数
  instance: 1
  instance_type: ecs.gn7i-c8g1.2xlarge
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/CVstudio/FaceChain_Agent.git
```