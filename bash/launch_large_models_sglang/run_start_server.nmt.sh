export HF_ENDPOINT=https://hf-mirror.com

path_to_config=../../model_configs/reward=wmt22-cometkiwi-da.json

root_to_save=./server_xcoment/
mkdir $root_to_save
#port=9000
#cuda=0


# GPU 总数和每个GPU启动的服务数
num_gpus=8
services_per_gpu=12

# 初始化端口号
port=9000

# 外层循环：遍历每个GPU
for gpu in $(seq 0 $((num_gpus - 1)))
do
  # 内层循环：为每个GPU启动指定数量的服务
  for service in $(seq 1 $services_per_gpu)
  do
    # 构建命令行
    cmd="nohup python ../../code/start_server.nmt_metric.py $path_to_config $gpu $root_to_save $port > log.nmt &"
    
    # 执行命令
    echo "Starting server on port $port with GPU $gpu"
    eval $cmd
    
    # 端口号递增
    ((port++))
  done
done

echo "All servers started."


