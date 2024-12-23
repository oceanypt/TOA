## mode
mode=ensemble_MoA

## I / O params
task=magpie
train_part_id=$1
data_name=Magpie-Pro-300K-Filtered.train.part_${train_part_id}.num=10000.jsonl #Magpie-Air-300K-Filtered.dev.num=100.jsonl
input=../../data/${task}/Magpie-Pro-300K-Filtered/$data_name
mkdir ../../output/$task/
save_mode='a'


###--> reward=ArmoRM.json
model_num=4
root_configs=./server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b_$2 #../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM
short_config_name=$config_name


## generation params
parallel_num=500
batch_size=10000

path_to_aggregator_prompt=../../prompts/aggregator.txt

n_iter=4
num_aggregation=${model_num}


for n_samples in $3 ; do
## sampling params
max_tokens=2048
temperature=0.7
top_p=1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.n_iter=${n_iter}.num_agg=${num_aggregation}.temp=${temperature}.top_p=${top_p}.jsonl

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
                                     --batch_size $batch_size \
                                     --path_to_aggregator_prompt $path_to_aggregator_prompt \
                                     --n_iter $n_iter \
                                     --num_aggregation $num_aggregation

done;
                                     
