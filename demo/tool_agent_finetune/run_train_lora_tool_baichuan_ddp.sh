DATA_PARALLEL_SIZE=2
TENSOR_MODEL_PARALLEL_SIZE=2

WORLD_SIZE=$(($DATA_PARALLEL_SIZE * $TENSOR_MODEL_PARALLEL_SIZE))
DATE_TIME=$(date +%Y%m%d-%H%M%S)
export PYTHONPATH=$PYTHONPATH:./
# torchrun examples/pytorch/my_llm_agent/finetune_tool.py \
torchrun --nproc_per_node $WORLD_SIZE demo/tool_agent_finetune/finetune_tool.py \
    --work_dir './tmp/tmp_baichuan/'$DATE_TIME \
    --model 'baichuan-inc/baichuan-7B' \
    --dataset_json_file 'demo/tool_agent_finetune/train_v2_plugins_sample.json' \
    --train_split 'train' \
    --val_split 'validation' \
    --max_epochs 1 \
    --per_device_train_batch_size 8 \
    --train_data_worker 0 \
    --lr 1e-4 \
    --lr_scheduler 'CosineAnnealingLR' \
    --bf16 1 \
    --device_map 'auto' \
    --train_shuffle 'True' \
    --save_best_checkpoint 'True' \
    --logging_interval 5 \
    --save_strategy 'by_step' \
    --save_interval 500 \
    --max_checkpoint_num 1 \
    --use_lora 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_replace_module 'pack' \
    --enable_gradient_checkpoint 1 \
    --max_length 2048 \
    --eval_strategy 'by_step' \
    --eval_interval 500 \
    --eval_metrics 'ppl' \
    --metric_for_best_model 'ppl' \
    --metric_rule_for_best_model 'min' \
    --per_device_eval_batch_size 8 \
    --eval_data_worker 0 \
    --max_checkpoint_num_best 1 \
    --deepspeed 'demo/tool_agent_finetune/default_offload_opt_param.json' \
    # --use_megatron 'True' \
    # --world_size $WORLD_SIZE \
    # --tensor_model_parallel_size $TENSOR_MODEL_PARALLEL_SIZE \

