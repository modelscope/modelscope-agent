CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --model_type qwen-7b \
    --sft_type lora \
    --ckpt_dir $1 \
    --dataset test_v2_plugins_sample.json \
    --dataset_sample -1 \
    --max_length 2048 \
    --dataset_test_size 1.0 \
