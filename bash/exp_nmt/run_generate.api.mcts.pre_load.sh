## mode
mode=ensemble_sample_N_MCTS

## I / O params
task=wmt22_test
data_name=test.cs_to_en.num=1448.jsonl 
source=$(echo $data_name | cut -d '.' -f 2 | cut -d '_' -f 1)
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'



###--> reward=ArmoRM.json
model_num=5
root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
root_reward_configs=../launch_large_models_sglang/server_xcoment/
config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
short_config_name=$config_name

parallel_num=250 #50 ## n_threads in MCTS
batch_size=1000

for n_samples in 32  ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

#MCTS
tau=1
alpha=0.05 
width=$(( model_num * n_samples / 3 ))
assert "[ $width -ge $model_num ]"

for topk_child in 1 ; do

echo $alpha

path_to_translation_template=../../prompts/translation.txt
path_to_refine_template=../../prompts/refinement_wo_feedback.nmt.v1.txt
output=../../output/${task}/${data_name}.source=${source}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.tau=${tau}.alpha=${alpha}.width=${width}.topk_child=${topk_child}.temp=${temperature}.top_p=${top_p}.temp_v=1.jsonl


python ../../code/ensemble_inference.nmt.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --root_reward_configs $root_reward_configs \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --tau $tau \
                                     --alpha $alpha \
                                     --parallel_num $parallel_num \
                                     --path_to_translation_template $path_to_translation_template \
                                     --path_to_refine_template $path_to_refine_template \
                                     --width $width \
                                     --topk_child $topk_child \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size \
                                     --source $source

                                
done
done
                                     
