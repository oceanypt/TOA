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


# port=9000
# cuda=0
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9001 
# cuda=0
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9002
# cuda=0
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9003
# cuda=1
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9004
# cuda=1
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9005
# cuda=1
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9006
# cuda=2
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9007
# cuda=2
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9008
# cuda=2
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9009
# cuda=3
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9010
# cuda=3
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9011
# cuda=3
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9012
# cuda=4
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9013
# cuda=4
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9014
# cuda=4
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9015
# cuda=5
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9016
# cuda=5
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9017
# cuda=5
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9018
# cuda=6
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &

# port=9019
# cuda=6
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9020
# cuda=6
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &



# port=9021
# cuda=7
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &


# port=9022
# cuda=7
# nohup python ../../code/start_server.nmt_metric.py  $path_to_config $cuda  $root_to_save  $port > log.nmt &
