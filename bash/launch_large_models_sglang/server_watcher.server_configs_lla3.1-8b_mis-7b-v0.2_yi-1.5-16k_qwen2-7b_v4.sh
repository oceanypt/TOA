#!/bin/bash

pip install modelscope
export HF_ENDPOINT=https://hf-mirror.com

# 间隔时间，以秒为单位
interval=900

config_dir_lla3=./server_configs_lla3.1-8b_v1
config_dir_mis_7b=./server_configs_mis_7b_v1
config_dir_yi=./server_configs_yi-1.5-16k_v1
config_dir_qwen2=./server_configs_qwen2-7b_v1

mkdir $config_dir_lla3
mkdir $config_dir_mis_7b
mkdir $config_dir_yi
mkdir $config_dir_qwen2


while true
do
    # kill
    bash kill_server.sh

    # 启动程序
    python start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json  $config_dir_mis_7b  0 8000 0.9 &

    python start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen2 1 8001 0.9  & 

    python start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir_lla3 2 8002 0.9  & 

    python start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir_yi 3 8003 0.9 & 

    python start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen2 4 8004 0.9 & 

    python start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir_mis_7b 5 8005 0.9 & 

    python start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir_lla3 6 8006 0.9  & 

    python start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir_yi 7 8007 0.9  & 

    # 等待指定的时间
    sleep $interval

    # 假设程序在运行时段内结束，我们需要重新启动它
    # 如果程序是持续运行的，这里可能需要杀掉旧的进程再启动新的
    # 可以使用 pkill, killall 等命令根据需要终止程序
    # 例如：pkill -f "$command"
done
