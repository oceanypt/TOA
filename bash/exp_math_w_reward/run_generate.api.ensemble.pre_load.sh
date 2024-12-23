pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False

## mode
mode=ensemble_sample_N

## I / O params
task=math
data_name=math.test.num=100.jsonl 
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'

# ###--> reward=ArmoRM.json
# model_num=5
# root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
# config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
# short_config_name=$config_name


###--> reward=GSM
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b #./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b  #../launch_large_models_sglang/$2 #server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_v3
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=GSM
short_config_name=$config_name

# ###--> reward=ArmoRM.json
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_lla-3.1-70b/
# root_reward_configs=../launch_large_models_sglang/server_xcoment/
# config_name=lla-3.1-70b.reward=ArmoRM
# short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_qwen2-7b_v2/
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0.json
config_name=qwen2-7b.reward=Qwen25_Math_RM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_v2/
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0.json
config_name=lla3.1-8b.reward=Qwen25_Math_RM
short_config_name=$config_name



###--> reward=Shepherd.json
model_num=2
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_qwen2-7b
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=lla3.1-8b_qwen2-7b.reward=Qwen25_Math_RM
short_config_name=$config_name


###--> reward=Shepherd.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_gemma-2-9b-it
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=gemma-2-9b-it.reward=Qwen25_Math_RM
short_config_name=$config_name


###--> reward=Shepherd.json
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_qwen2-7b_gemma-2-9b_deepseek_math_7b_rl
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=lla3.1-8b_qwen2-7b_gemma-2-9b_deepseek_math_7b_rl.reward=Qwen25_Math_RM
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


# ###--> reward=Shepherd.json
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_$1
# path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
# config_name=${1}.reward=Qwen25_Math_RM
# short_config_name=$config_name

## generation params
batch_size=100000  # the number of data for generation and save for one generation
parallel_num=25 # how many data points for generation for one time


for n_samples in $1 ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl


# python ../../code/ensemble_inference.math.server_pre_load.fast.py --mode $mode \
#                                      --input $input \
#                                      --output $output \
#                                      --root_configs $root_configs \
#                                      --n_samples $n_samples \
#                                      --max_tokens $max_tokens \
#                                      --temperature $temperature \
#                                      --top_p $top_p \
#                                      --parallel_num $parallel_num \
#                                      --save_mode $save_mode \
#                                      --batch_size $batch_size 

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
                                     
