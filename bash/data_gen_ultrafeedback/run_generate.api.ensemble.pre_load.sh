## mode
mode=ensemble_sample_N  # parallel ensemble or random sampling

## I / O params
task=ultra_feedback
data_name=train.ultrafeedback.num=59876.jsonl
input=../../data/${task}/$data_name
mkdir ../../output/$task/
save_mode='a'


# ## model params
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
path_reward_config=../../model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
echo $path_reward_config
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM
short_config_name=$config_name

# ## model params
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_qwen2-7b/
# path_reward_config=../../model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
# config_name=qwen2-7b.reward=ArmoRM
# short_config_name=$config_name

# ## model params
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b/
# path_reward_config=../../model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
# config_name=lla3.1-8b.reward=ArmoRM
# short_config_name=$config_name



## generation params
parallel_num=100 
batch_size=10000  


for n_samples in 40 ; do  ## to be noted, each input question will generate [ n_samples * model_num ] responses

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl


python ../../code/ensemble_inference.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --path_reward_config $path_reward_config \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size

done;
                                     
