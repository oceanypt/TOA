config_dir=server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v3

for N in  16 32 48 64 80 96 112 128  ; do

nohup bash run_generate.api.moa.pre_load.sh $N  $config_dir  > log.moa & 
nohup bash run_generate.api.ensemble.pre_load.sh $N  $config_dir > log.ensemble &
nohup bash run_generate.api.ensemble_seq.pre_load.sh $N $config_dir > log.ensemble_seq & wait

done