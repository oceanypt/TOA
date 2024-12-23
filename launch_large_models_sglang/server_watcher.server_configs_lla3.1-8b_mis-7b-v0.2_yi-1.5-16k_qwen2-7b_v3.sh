#!/bin/bash

pip install modelscope
export HF_ENDPOINT=https://hf-mirror.com

# 间隔时间，以秒为单位
interval=900

config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v3
mkdir config_dir

while true
do
    # kill
    bash kill_server.sh

    # 启动程序
    python start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 0 8000 0.9 &

    python start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 1 8001 0.9  & 

    python start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 2 8002 0.9  & 

    python start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 3 8003 0.9 & 

    python start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 4 8004 0.9 & 

    python start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 5 8005 0.9 & 

    python start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 6 8006 0.9  & 

    python start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 7 8007 0.9  & 

    # 等待指定的时间
    sleep $interval

    # 假设程序在运行时段内结束，我们需要重新启动它
    # 如果程序是持续运行的，这里可能需要杀掉旧的进程再启动新的
    # 可以使用 pkill, killall 等命令根据需要终止程序
    # 例如：pkill -f "$command"
done
