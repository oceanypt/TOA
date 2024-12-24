## mode
mode=ensemble_sample_N

## I / O params
task=wmt22_test
data_name=test.zh_to_en.num=200.jsonl 
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


## generation params
batch_size=1000  
parallel_num=500 

for n_samples in 32 ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

path_to_translation_template=../../prompts/translation.txt

output=../../output/$task/${data_name}.source=${source}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl


python ../../code/ensemble_inference.nmt.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --path_to_translation_template $path_to_translation_template \
                                     --root_reward_configs $root_reward_configs \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size \
                                     --source $source

done;
                                     
