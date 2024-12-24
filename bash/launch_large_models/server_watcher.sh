# #!/bin/bash
# pip install modelscope
# export HF_ENDPOINT=https://hf-mirror.com

# # 间隔时间，以秒为单位
# interval=2000

# config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
# config_dir_mis=./server_configs_mis-7b-v0.2
# config_dir_lla=./server_configs_lla3.1-8b
# config_dir_yi=./server_configs_yi-1.5-16k
# config_dir_qwen=./server_configs_qwen2-7b

# mkdir $config_dir
# mkdir $config_dir_mis
# mkdir $config_dir_lla
# mkdir $config_dir_qwen
# mkdir $config_dir_yi


# root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

# while true
# do
#     # kill
#     bash kill_server.sh

#     # 启动程序
#     python $root_dir/start_server.vllm.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir_mis 0 8000 0.9  &

#     python $root_dir/start_server.vllm.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 1 8001 0.9   &


#     python $root_dir/start_server.vllm.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir_lla 2 8002 0.9  &

#     python $root_dir/start_server.vllm.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir_yi 3 8003 0.9 &


#     python $root_dir/start_server.vllm.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 4 8004 0.9    &


#     python $root_dir/start_server.vllm.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir_mis 5 8005 0.9  &


#     python $root_dir/start_server.vllm.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir_lla 6 8006 0.9 &


#     python $root_dir/start_server.vllm.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir_yi 7 8007 0.9 &


#     sleep 20

#     cp $config_dir_mis/*  $config_dir/
#     cp $config_dir_lla/*  $config_dir/
#     cp $config_dir_qwen/*  $config_dir/
#     cp $config_dir_yi/*  $config_dir/


#     # 等待指定的时间
#     sleep $interval
#     # 假设程序在运行时段内结束，我们需要重新启动它
#     # 如果程序是持续运行的，这里可能需要杀掉旧的进程再启动新的
#     # 可以使用 pkill, killall 等命令根据需要终止程序
#     # 例如：pkill -f "$command"
# done

# #!/bin/bash
# pip install modelscope
# export HF_ENDPOINT=https://hf-mirror.com

# # 间隔时间，以秒为单位
# interval=3600
# config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
# config_dir_mis=./server_configs_mis-7b-v0.2
# config_dir_lla=./server_configs_lla3.1-8b
# config_dir_yi=./server_configs_yi-1.5-16k
# config_dir_qwen=./server_configs_qwen2-7b

# mkdir $config_dir
# mkdir $config_dir_mis
# mkdir $config_dir_lla
# mkdir $config_dir_qwen
# mkdir $config_dir_yi

# #rm $config_dir/*
# #rm $config_dir_mis/*
# #rm $config_dir_lla/*
# #rm $config_dir_qwen/*
# #rm $config_dir_yi/*


# root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

# while true
# do
#     # kill
#     bash kill_server.sh

#     # 启动程序

#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 0 8000 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 1 8001 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 2 8002 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 3 8003 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 4 8004 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 5 8005 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 6 8006 0.9   &
#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 7 8007 0.9   &

#     sleep 20

#     cp $config_dir_qwen/*  $config_dir/

#     # 等待指定的时间
#     sleep $interval
#     # 假设程序在运行时段内结束，我们需要重新启动它
#     # 如果程序是持续运行的，这里可能需要杀掉旧的进程再启动新的
#     # 可以使用 pkill, killall 等命令根据需要终止程序
#     # 例如：pkill -f "$command"
# done




# #!/bin/bash
# pip install modelscope
# export HF_ENDPOINT=https://hf-mirror.com

# # 间隔时间，以秒为单位
# interval=1500

# config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v2

# mkdir $config_dir

# root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

# while true
# do
#     # kill
#     bash kill_server.sh

#     # 启动程序
#     python $root_dir/start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 0 8000 0.9  &

#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 1 8001 0.9   &


#     python $root_dir/start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 2 8002 0.9  &

#     python $root_dir/start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 3 8003 0.9 &


#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 4 8004 0.9    &


#     python $root_dir/start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 5 8005 0.9  &


#     python $root_dir/start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 6 8006 0.9 &


#     python $root_dir/start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 7 8007 0.9 &


#     sleep 20


#     # 等待指定的时间
#     sleep $interval
#     # 假设程序在运行时段内结束，我们需要重新启动它
#     # 如果程序是持续运行的，这里可能需要杀掉旧的进程再启动新的
#     # 可以使用 pkill, killall 等命令根据需要终止程序
#     # 例如：pkill -f "$command"
# done




# #!/bin/bash
# pip install modelscope
# export HF_ENDPOINT=https://hf-mirror.com

# # 间隔时间，以秒为单位
# interval=3000

# config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v2
# config_dir_mis=./server_configs_mis-7b-v0.2_v2
# config_dir_lla=./server_configs_lla3.1-8b_v2
# config_dir_yi=./server_configs_yi-1.5-16k_v2
# config_dir_qwen=./server_configs_qwen2-7b_v2

# config_dir_qwen_lla=./server_configs_lla3.1-8b_qwen2-7b


# mkdir $config_dir
# mkdir $config_dir_mis
# mkdir $config_dir_lla
# mkdir $config_dir_qwen
# mkdir $config_dir_yi
# mkdir $config_dir_qwen_lla

# # rm $config_dir/*
# # rm $config_dir_mis/*
# # rm $config_dir_lla/*
# # rm $config_dir_qwen/*
# # rm $config_dir_yi/*
# # rm $config_dir_qwen_lla/*


# root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

# while true
# do
#     # kill
#     bash kill_server.sh

#     # 启动程序
#     python $root_dir/start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir_mis 0 8000 0.9  &

#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 1 8001 0.9   &


#     python $root_dir/start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir_lla 2 8002 0.9  &

#     python $root_dir/start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir_yi 3 8003 0.9 &


#     python $root_dir/start_server.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir_qwen 4 8004 0.9    &


#     python $root_dir/start_server.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir_mis 5 8005 0.9  &


#     python $root_dir/start_server.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir_lla 6 8006 0.9 &


#     python $root_dir/start_server.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir_yi 7 8007 0.9 &


#     sleep 20

#     cp $config_dir_mis/*  $config_dir/
#     cp $config_dir_lla/*  $config_dir/
#     cp $config_dir_qwen/*  $config_dir/
#     cp $config_dir_yi/*  $config_dir/


#     cp $config_dir_lla/* $config_dir_qwen_lla/
#     cp $config_dir_qwen/* $config_dir_qwen_lla/

#     # 等待指定的时间
#     sleep $interval
#     # 假设程序在运行时段内结束，我们需要重新启动它
#     # 如果程序是持续运行的，这里可能需要杀掉旧的进程再启动新的
#     # 可以使用 pkill, killall 等命令根据需要终止程序
#     # 例如：pkill -f "$command"
# done





# #!/bin/bash
# pip install modelscope
# export HF_ENDPOINT=https://hf-mirror.com

# # 间隔时间，以秒为单位
# interval=1200

# root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

# while true
# do
#     # kill
#     bash kill_server.sh
    
#     # 启动程序
#     python $root_dir/start_server.sglang.py ../../model_configs/llama-3.1-70b-instruct.json server_configs_llama-3.1-70b-instruct 0,1,2,3,4,5,6,7 8000 0.85

#     # 等待指定的时间
#     sleep $interval
# done





#!/bin/bash
pip install modelscope
export HF_ENDPOINT=https://hf-mirror.com

# 间隔时间，以秒为单位
interval=2000 #2000

config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b

root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

while true
do
    # kill
    bash kill_server.sh

    # 启动程序
    python $root_dir/start_server.vllm.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 0 8000 0.9  &

    python $root_dir/start_server.vllm.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 1 8001 0.9   &


    python $root_dir/start_server.vllm.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 2 8002 0.9  &

    python $root_dir/start_server.vllm.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 3 8003 0.9 &


    python $root_dir/start_server.vllm.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 4 8004 0.9    &


    python $root_dir/start_server.vllm.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 5 8005 0.9  &


    python $root_dir/start_server.vllm.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 6 8006 0.9 &


    python $root_dir/start_server.vllm.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 7 8007 0.9 &


    # 等待指定的时间
    sleep $interval
done




# #!/bin/bash
# pip install modelscope
# export HF_ENDPOINT=https://hf-mirror.com

# # 间隔时间，以秒为单位
# interval=432000 #2000

# config_dir=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b

# root_dir=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/bash/launch_large_models_sglang/

# while true
# do
#     # kill
#     bash kill_server.sh

#     # 启动程序
#     python $root_dir/start_server.sglang.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 0 8000 0.9  &

#     python $root_dir/start_server.sglang.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 1 8001 0.9   &


#     python $root_dir/start_server.sglang.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 2 8002 0.9  &

#     python $root_dir/start_server.sglang.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 3 8003 0.9 &


#     python $root_dir/start_server.sglang.py ../../model_configs/Qwen2-7B-Instruct.json $config_dir 4 8004 0.9    &


#     python $root_dir/start_server.sglang.py ../../model_configs/Mistral-7B-Instruct-v0.2.json $config_dir 5 8005 0.9  &


#     python $root_dir/start_server.sglang.py ../../model_configs/llama-3.1-8b-instruct.json $config_dir 6 8006 0.9 &


#     python $root_dir/start_server.sglang.py ../../model_configs/Yi-1.5-9B-Chat-16K.json $config_dir 7 8007 0.9 &


#     # 等待指定的时间
#     sleep $interval
# done