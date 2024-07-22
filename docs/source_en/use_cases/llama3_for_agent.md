# Llama3 Chinese General Agent Fine-Tuned Model
Since the Llama3 model was announced on April 18, Chinese developers have conducted extensive training and adaptation on the Llama3 model. In addition to the Chinese pure text model, multi-modal versions are also being released.

## Introduction
Since the Llama3 model was announced on April 18, Chinese developers have conducted extensive training and adaptation on the Llama3 model. In addition to the Chinese pure text model, multi-modal versions are also being released. Considering the demand for Agent scenarios among domestic users, Modelscope Community's LLM&AIGC model fine-tuning and inference framework SWIFT has trained a general Chinese model based on the original Llama3-8b-instruct version, retaining and adapting Chinese Agent capabilities. This is the first general Agent Llama3 model in the open-source community that is fully adapted to the Chinese environment, with a more comprehensive evaluation report to follow.

[Model Link](https://modelscope.cn/models/swift/Llama3-Chinese-8B-Instruct-Agent-v1/summary)

## Usage
It is recommended for users to use swift for direct inference or deployment:

```shell
# Install dependencies
pip install ms-swift -U
```

```shell
# Inference
swift infer --model_type llama3-8b-instruct --model_id_or_path swift/Llama3-Chinese-8B-Instruct-Agent-v1
```

```shell
# Deployment
swift deploy --model_type llama3-8b-instruct --model_id_or_path swift/Llama3-Chinese-8B-Instruct-Agent-v1
```

This model can be used with the ModelScope-Agent framework. Please refer to:
[Using with ModelScopeAgent Framework](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Agent%E5%BE%AE%E8%B0%83%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md#%E6%90%AD%E9%85%8Dmodelscope-agent%E4%BD%BF%E7%94%A8)

We also encourage developers to perform secondary fine-tuning on this model and future versions like v2 or v3 to achieve better capabilities.

Below is an introduction on how to train a Llama3 Chinese Agent model using the SWIFT framework.

## Environment Setup
We used Modelscope's official framework SWIFT for model training: [Jump to here](https://github.com/modelscope/swift/tree/main). Developers who want to train a Chinese version of Llama3 can refer to the following installation method:

```shell
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
# Align environment (usually not needed. If you encounter errors, you can run the following commands; the repository uses the latest environment for testing)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## Data Preparation
To adapt to Chinese and Agent scenarios, we mixed the corpus in a certain ratio. The corpus used to train Llama3 includes:

- [COIG-CQIA](https://modelscope.cn/datasets/AI-ModelScope/COIG-CQIA/summary) This dataset contains Chinese internet information from sources like traditional Chinese knowledge, Douban, Zhihu, etc.
- [Modelscope General Agent Training Dataset](https://modelscope.cn/datasets/AI-ModelScope/ms-agent-for-agentfabric/summary)
- [alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)
- [ms-bench Modelscope General Chinese Q&A Dataset](https://modelscope.cn/datasets/iic/ms_bench/summary)

SWIFT supports many other open-source datasets beneficial for training, such as:
- Firefly Chinese Dataset
- DeepCtrl Multi-lingual Dataset
- Alpaca/ShareGPT

If developers wish to use other datasets to train Llama3, it is as simple as specifying `--dataset firefly-all-zh` in the command line to utilize them. The complete list of supported datasets can be found at:
[Complete List of Supported Datasets](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%95%B0%E6%8D%AE%E9%9B%86)

We have added MLP and Embedder to the `lora_target_modules`. You can add lora to all linear layers (including qkvo, mlp, and embedder) by specifying `--lora_target_modules ALL`. This usually provides the best results.

| Hyperparameter                | Value    |
| ----------------------------- | -------- |
| lr                            | 5e-5     |
| epoch                         | 2        |
| lora_rank                     | 8        |
| lora_alpha                    | 32       |
| lora_target_modules           | ALL      |
| batch_size                    | 2        |
| gradient_accumulation_steps   | 16       |

Training was conducted using 8 GPUs. After environment setup is complete, you can start training with the following command:

```shell
NPROC_PER_NODE=8 \
swift sft \
  --model_type llama3-8b-instruct \
  --dataset ms-agent-for-agentfabric-default alpaca-en ms-bench ms-agent-for-agentfabric-addition coig-cqia-ruozhiba coig-cqia-zhihu coig-cqia-exam coig-cqia-chinese-traditional coig-cqia-logi-qa coig-cqia-segmentfault coig-cqia-wiki \
  --batch_size 2 \
  --max_length 2048 \
  --use_loss_scale true \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --use_flash_attn true \
  --eval_steps 500 \
  --save_steps 500 \
  --train_dataset_sample -1 \
  --dataset_test_ratio 0.1 \
  --val_dataset_sample 10000 \
  --num_train_epochs 2 \
  --check_dataset_strategy none \
  --gradient_checkpointing true \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --sft_type lora \
  --lora_target_modules ALL \
  --lora_rank 8 \
  --lora_alpha 32
```

To improve the accuracy of the ReAct format, we increased the weight of some loss fields to retain agent capability performance in Chinese training.
The trained model can be downloaded from the Modelscope official website: [Download Link](https://modelscope.cn/models/swift/Llama3-Chinese-8B-Instruct-Agent-v1/summary)

## Inference Results
This model has excellent Chinese Q&A capabilities, as shown in the following examples:

General Q&A:
![General Answer 1](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_48028fa857ce4308ba14ef80e8bb1952.png)
![General Answer 2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_cf86fb4cffa14a1d956c6d5d15bfda2c.png)

Logic Problems:
![Logic Problem 1](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_0d8ccc38583a491a9d060a8e7d87ca07.png)
![Logic Problem 2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_4db5a56965ee409a9b3324809b22264a.png)
![Logic Problem 3](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_3dfc1312acd549398c59dc2b6a9c9983.png)

Couplets:
![Couplets](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_5be99d8ef7704354bf67c649d9e09b8d.png)

Acrostic Poem:
![Acrostic Poem](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_c6368828deec4d06a111ebba06cbe757.png)

Classical Chinese Translation:
![Classical Chinese Translation](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_3bae3959af4647fcab3c0707e1d9f15f.png)

Coding Capability:
![Coding Capability](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_3d2b0b206d0f42b29f339fc7829ff1e3.png)

## Evaluation
We used the `swift eval` command to evaluate the general capabilities of the model before and after training. The results are as follows:

| Model Evaluation          | ARC     | CEVAL   | GSM8K   |
| ------------------------- | ------- | ------- | ------- |
| Llama3-8b-instruct         | 0.7645 | 0.5089 | 0.7475 |
| Llama3-Chinese-8B-Instruct-Agent-v1 | 0.7577 | 0.4903 | 0.652  |

The GSM8K capability in English decreased by about 8 points. Through ablation experiments, we found that removing the alpaca-en corpus results in a drop of at least ten points in GSM8K.

Developers can also use the swift framework to evaluate other models. The command is very simple:

```shell
swift eval --model_type llama3-8b-instruct --model_id_or_type LLM-Research/Meta-Llama-3-8B-Instruct --infer_backend pt --eval_dataset ceval arc
```

## Usage with ModelScope-Agent
For usage with ModelScope-Agent, please refer to our official documentation: [Official Documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Agent%E5%BE%AE%E8%B0%83%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md#%E5%9C%A8%E5%91%BD%E4%BB%A4%E8%A1%8C%E4%B8%AD%E4%BD%BF%E7%94%A8agent)

After deploying the service, we can verify its API call effectiveness in AgentFabric. For example, with weather queries, you can see:

![Result 1](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_60c97d99bfe94c399b5b325e2c8f01be.png)
![Result 2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_691859d448464f129a166c59ecb6ff29.png)

The model can complete the query as required by the system.

**Text-to-Image**
![Text-to-Image](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_bcfbb71303224e1d8bcb1ce74aa23878.png)

**Image Explanation**
![Input Image](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_22596052e9344af0897d88d3c25e264f.png)
![Return Result 1](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_0db1a3d819a04d3abdda3d70af296d1d.png)
![Return Result 2](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_afb4bf7b772e436db83ee7dbde29ca06.png)
![Return Result 3](https://ucc.alicdn.com/pic/developer-ecology/umvm3uqpbgldm_9b065388cb274c5db157191c9e1a7a17.png)

## Areas for Improvement
1. The original Llama3 English model has some CoT capabilities. When training for Chinese, some knowledge was lost. This issue will continue to be addressed in the V2 version.
2. The proportion of English corpus needs adjustment to ensure the original English capabilities (such as sensitive metrics like GSM8K).
