config_dir_lla3=./server_configs_lla3.1-8b_v1
config_dir_mis_7b=./server_configs_mis_7b_v1
config_dir_yi=./server_configs_yi-1.5-16k_v1
config_dir_qwen2=./server_configs_qwen2-7b_v1


for N in  64 128 192 256 320 384 448 512 ; do

nohup bash run_generate.api.single.sh $config_dir_lla3  $N  > log.single & 
nohup bash run_generate.api.single.sh $config_dir_mis_7b  $N  > log.single & 
nohup bash run_generate.api.single.sh $config_dir_yi  $N  > log.single & 
nohup bash run_generate.api.single.sh $config_dir_qwen2  $N  > log.single & 

wait

done