# 训练

我们在ModelScope发布了一个工具数据集，用于LLM的微调训练和评估（[工具数据集](https://www.modelscope.cn/datasets/modelscope/ms_hackathon_23_agent_train_dev/summary)）。 提供了对应的基于 **ModelScope Library** 的训练脚本。

## 训练选项

训练脚本支持多种训练方法，可根据您的可用资源进行选择：

- 使用全参数或 Lora 进行微调。
- 使用集成的 ModelScope DeepspeedHook 进行分布式训练。

## 数据预处理

一条数据的格式如下所示。 由于我们使用文本生成任务方案来训练LLM，因此需要对原始数据进行预处理。

```Python
System: system info(agent info, tool info...).
User: user inputs.
Assistants:
# agent call
<|startofthink|>...<|endofthink|>\n\n
# tool execute
<|startofexec|>...<|endofexec|>\n
# summarize
...
# may be multiple rounds
```

- 每个数据实例由三个角色组成：system、user和assistant。 LLM应该只关注**assistant**部分。
- **assistant**部分通常由三个部分组成。 LLM应该只考虑agent通话的内容和最终总结。
- 其他不必要的部分使用`IGNORE_INDEX`进行屏蔽，以将它们排除在损失计算之外。


## 训练脚本

使用脚本`run_train_ddp.sh`拉起训练。

```Shell
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_type modelscope-agent-7b \
    --sft_type lora \
    --output_dir runs \
    --dataset damo/MSAgent-Bench \
    --dataset_sample 20000 \
    --dataset_test_ratio 0.02 \
    --max_length 2048 \
    --dtype bf16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 20 \
    --use_flash_attn true \
```

## 评估

训练结束后，我们还提供了一个评估脚本来判断 agent 在测试数据集上的表现。 测试数据集的 ground truth 来自于。。。。。。

评估指标包括：
- 工具名称和参数的准确率。
- 摘要相似度指标`Rouge-l`。
