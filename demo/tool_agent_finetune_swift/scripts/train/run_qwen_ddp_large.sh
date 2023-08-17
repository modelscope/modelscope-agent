nproc_per_node=4
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_type qwen-7b \
    --sft_type lora \
    --output_dir runs \
    --dataset train_v2_plugins_sample.json \
    --dataset_sample -1 \
    --max_length 2048 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --ddp_backend nccl \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \