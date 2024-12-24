## mode
mode=ensemble_MoA

## I / O params
task=alpaca_eval
data_name=alpaca_eval.num=805.jsonl 
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'


###--> reward=ArmoRM.json
model_num=5
root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
path_reward_config=../../model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=0.json
config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
short_config_name=$config_name


## generation params
batch_size=1000
parallel_num=500

path_to_aggregator_prompt=../../prompts/aggregator.txt

n_iter=4
num_aggregation=${model_num}


for n_samples in 32  ; do
## sampling params
max_tokens=2048
temperature=0.7
top_p=1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.n_iter=${n_iter}.num_agg=${num_aggregation}.temp=${temperature}.top_p=${top_p}.jsonl

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
                                     