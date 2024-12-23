## mode
mode=ensemble_MoA

# ## I / O params
# task=data_gen_nmt
# data_name=dev.zh_to_en.num=100.jsonl  #alpaca_eval.num=805.part_1.num=100.jsonl #alpaca_eval.num=30.jsonl
# source=zh
# input=../../data/$task/$data_name
# mkdir ../../output/$task/
# save_mode='a'

## I / O params
task=wmt22_test
data_name=$2 #test.zh_to_en.num=200.jsonl #dev.zh_to_en.num=100.jsonl  #alpaca_eval.num=805.part_1.num=100.jsonl #alpaca_eval.num=30.jsonl
source=$(echo $data_name | cut -d '.' -f 2 | cut -d '_' -f 1) #zh
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'


## model params
###--> reward=ArmoRM.json
model_num=5
root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
root_reward_configs=../launch_large_models_sglang/server_xcoment/
config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
short_config_name=$config_name


## generation params
parallel_num=500
save_mode='a'
batch_size=1000

n_iter=3
num_aggregation=${model_num}

for n_samples in $1 ; do
## sampling params
max_tokens=2048
temperature=0.7
top_p=1

path_to_translation_template=../../prompts/translation.txt
path_to_refine_template=../../prompts/aggregator.nmt.v2.txt #refinement_wo_feedback.nmt.v1.txt

output=../../output/$task/${data_name}.source=${source}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.n_iter=${n_iter}.num_agg=${num_aggregation}.temp=${temperature}.top_p=${top_p}.agg=v2.jsonl

#CUDA_VISIBLE_DEVICES=$GPU 
python ../../code/ensemble_inference.nmt.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --root_reward_configs $root_reward_configs \
                                     --n_samples $n_samples \
                                     --path_to_translation_template $path_to_translation_template \
                                     --path_to_refine_template $path_to_refine_template \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size \
                                     --n_iter $n_iter \
                                     --num_aggregation $num_aggregation \
                                     --source $source

done;
                                     
