export PYTHONPATH=./

export VLLM_USE_MODELSCOPE=True
python -m vllm.entrypoints.openai.api_server \
 --model=iic/alpha-umi-planner-7b \
 --revision=v1.0.0 --trust-remote-code \
 --port 8090 \
 --dtype float16 \
 --gpu-memory-utilization 0.3 > planner.log &

python -m vllm.entrypoints.openai.api_server \
 --model=iic/alpha-umi-caller-7b \
 --revision=v1.0.0 --trust-remote-code \
 --port 8091 \
 --dtype float16 \
 --gpu-memory-utilization 0.3 > caller.log &

python -m vllm.entrypoints.openai.api_server \
 --model=iic/alpha-umi-summarizer-7b \
 --revision=v1.0.0 --trust-remote-code \
 --port 8092 \
 --dtype float16 \
 --gpu-memory-utilization 0.3 > summarizer.log &
