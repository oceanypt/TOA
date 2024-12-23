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
# task=alpaca_eval_arena_hard
# data_name=alpaca_eval.num=805.jsonl  #alpaca_eval.num=805.part_1.num=100.jsonl #alpaca_eval.num=30.jsonl
# input=../../data/$task/$data_name
# mkdir ../../output/$task/
# save_mode='a'

## I / O params
task=magpie
train_part_id=$1
data_name=Magpie-Pro-300K-Filtered.train.part_${train_part_id}.num=10000.jsonl #Magpie-Air-300K-Filtered.dev.num=100.jsonl
input=../../data/${task}/Magpie-Pro-300K-Filtered/$data_name
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
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b #./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_$1 #../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=${1}.json
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM
short_config_name=$config_name


parallel_num=$3 #500 #50 ## n_threads in MCTS
batch_size=10000

for n_samples in $2  ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

#MCTS
tau=0.1 #0.1
alpha=0.01
width=$(( model_num * n_samples / 3 ))
assert "[ $width -ge $model_num ]"

for topk_child in 1 ; do

echo $alpha

path_to_refine_template=../../prompts/refinement_wo_feedback.inst_following.v13.txt
output=../../output/${task}/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.tau=${tau}.alpha=${alpha}.width=${width}.topk_child=${topk_child}.temp=${temperature}.top_p=${top_p}.temp_v=13.jsonl


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
                                     
