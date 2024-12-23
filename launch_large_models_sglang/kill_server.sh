pkill -9 -f "python -m sglang.launch_server"
pkill -9 -f "/root/anaconda3/envs/sglang/bin/python"
pkill -9 -f "python -m vllm.entrypoints.openai.api_server"
pkill -9 -f "/root/anaconda3/envs/vllm/bin/python"

pkill -9 -f "/root/anaconda3/envs/llama31_infer/bin/python -c from multiprocessing"
