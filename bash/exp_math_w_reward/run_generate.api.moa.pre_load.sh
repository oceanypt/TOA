## mode
mode=ensemble_MoA

## I / O params
task=math
data_name=math.test.num=100.jsonl
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'


# ## model params
# ###--> reward=ArmoRM.json
# model_num=5
# root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
# root_reward_configs=../launch_large_models_sglang/server_xcoment/
# config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
# short_config_name=$config_name

###--> reward=GSM
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b  #server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v3 #./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b #../launch_large_models_sglang/$2 #server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v3
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=GSM
short_config_name=$config_name


###--> reward=Shepherd.json
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=Qwen25_Math_RM
short_config_name=$config_name


###--> reward=Shepherd.json
model_num=2
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_qwen2-7b
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=lla3.1-8b_qwen2-7b.reward=Qwen25_Math_RM
short_config_name=$config_name



## generation params
parallel_num=25
save_mode='a'
batch_size=1000000

n_iter=3
num_aggregation=${model_num}

for n_samples in $1 ; do
## sampling params
max_tokens=2048
temperature=0.7
top_p=1

path_to_aggregator_prompt=../../prompts/aggregator.math.v1.txt

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.n_iter=${n_iter}.num_agg=${num_aggregation}.temp=${temperature}.top_p=${top_p}.agg=math_v1.jsonl

#CUDA_VISIBLE_DEVICES=$GPU 
python ../../code/ensemble_inference.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --n_samples $n_samples \
                                     --path_to_aggregator_prompt $path_to_aggregator_prompt \
                                     --path_reward_config $path_reward_config \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size \
                                     --n_iter $n_iter \
                                     --num_aggregation $num_aggregation

done;
                                     
