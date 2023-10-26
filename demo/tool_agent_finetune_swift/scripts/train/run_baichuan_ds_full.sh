nproc_per_node=2
# PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft_new.py \
    --model_type baichuan2-7b \
    --sft_type full \
    --template_type baichuan \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset data/train_v1.4_plugins_story_face_agent_v2.json \
    --train_dataset_sample 2000 \
    --num_train_epochs 1 \
    --max_length 128\
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0. \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --push_to_hub false \
    --hub_model_id baichuan2-7b-chat-lora \
    --hub_private_repo true \
    --hub_token 'your-sdk-token' \
    --deepspeed_config_path 'scripts/train/ds_stage_2.json' \
    --only_save_model true \