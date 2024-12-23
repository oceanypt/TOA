pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False

## mode
mode=ensemble_sample_N

## I / O params
task=alpaca_eval_arena_hard
data_name=alpaca_eval.num=805.jsonl #alpaca_eval.num=805.part_1.num=100.jsonl #alpaca_eval.num=30.jsonl
input=../../data/$task/$data_name
mkdir ../../output/$task/

## model params
###--> reward=ArmoRM.json
# model_num=5
# root_configs=../launch_large_models_sglang/server_configs_num=5/
# path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=0.json
# config_name=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
# short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=5
root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
short_config_name=$config_name



###--> reward=ArmoRM.json
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=2.json
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_ultrafeedback_Qwen2-7B-Instruct_sft_data=single_qwn2_7b
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=2.json
config_name=ultrafeedback_Qwen2-7B-Instruct_sft_data=single_qwn2_7b.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_lla3-8b-instruct
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=0.json
config_name=lla3-8b-instruct.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_qwen2-7b-instruct
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=0.json
config_name=qwen2-7b-instruct.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_ultrafeedback_Qwen2-7B-Instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30
#path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=2.json
path_reward_config=../../model_configs/emptyRM.json
config_name=ultrafeedback_Qwen2-7B-Instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_yi-1.5-16k
#path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=2.json
path_reward_config=../../model_configs/emptyRM.json
config_name=yi-1.5-16k.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_ultrafeedback_llama3-8b-instruct_sft_data=seq_small
#path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=2.json
path_reward_config=../../model_configs/emptyRM.json
config_name=ultrafeedback_llama3-8b-instruct_sft_data=seq_small.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_ultrafeedback_Qwen2-7B-Instruct_sft_data=seq_small
#path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=2.json
path_reward_config=../../model_configs/emptyRM.json
config_name=ultrafeedback_Qwen2-7B-Instruct_sft_data=seq_small.reward=ArmoRM
short_config_name=$config_name





## generation params
batch_size=1000  # the number of data for generation and save for one generation
parallel_num=805 # how many data points for generation for one time
save_mode='a'

for n_samples in $1 ; do

## sampling params
max_tokens=2048
temperature=0.7 #0.7 #0.001 #0.7 #0.001 #0.001 #0.001 #0.7 #0.7
top_p=1 #1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl
#output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.concise_v2.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl


#CUDA_VISIBLE_DEVICES=$GPU
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
                                     
