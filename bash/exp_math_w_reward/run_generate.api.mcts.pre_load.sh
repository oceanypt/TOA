assert() {
    # "$@" 执行传入的命令，这里是条件判断
    if ! eval $@; then
        echo "Assertion failed: $@"
        exit 1
    fi
}
pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False
## mode
mode=ensemble_sample_N_MCTS

## I / O params
task=math
data_name=math.test.num=100.jsonl 
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'

# ###--> reward=ArmoRM.json
# model_num=5
# root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
# path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
# config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
# short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=0.json
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM
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


###--> reward=Shepherd.json
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_qwen2-7b_gemma-2-9b_deepseek_math_7b_rl
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=lla3.1-8b_qwen2-7b_gemma-2-9b_deepseek_math_7b_rl.reward=Qwen25_Math_RM
short_config_name=$config_name


###--> reward=Shepherd.json
model_num=2
root_configs=../launch_large_models_sglang/server_configs_qwen2-7b_gemma-2-9b
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=qwen2-7b_gemma-2-9b.reward=Qwen25_Math_RM
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

parallel_num=25 #50 ## n_threads in MCTS
batch_size=1000


#for n_samples in 8 16 24 32 40 48 56 ; do
#for n_samples in 96 160 224  288 ; do
for n_samples in $1 ; do


## sampling params
max_tokens=2048
temperature=0.7
top_p=1

#MCTS
tau=1 #0.5 #0.1
alpha=0.1 #0.1 #0.05 #0.01
#width=$(( model_num * n_samples / 3 ))
width=$(( model_num * n_samples / 2 ))
assert "[ $width -ge $model_num ]"



for topk_child in 1 ; do

echo $alpha

path_to_refine_template=../../prompts/refinement_wo_feedback.math.v1.txt
output=../../output/${task}/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.tau=${tau}.alpha=${alpha}.width=${width}.topk_child=${topk_child}.temp=${temperature}.top_p=${top_p}.temp_v=math_v1.jsonl


python ../../code/ensemble_inference.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --path_reward_config $path_reward_config \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --tau $tau \
                                     --alpha $alpha \
                                     --parallel_num $parallel_num \
                                     --path_to_refine_template $path_to_refine_template \
                                     --width $width \
                                     --topk_child $topk_child \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size

                                
done
done
                                     
