## mode
mode=ensemble_sample_N_MCTS


## I / O params
# - specifiy the input data path
task=ultra_feedback
data_name=train.ultrafeedback.num=59876.jsonl

input=../../data/${task}/$data_name
mkdir ../../output/$task/
save_mode='a'

## model configuration path
model_num=4
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b # policy model
path_reward_config=../../model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json  # reward model

config_name=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM
short_config_name=$config_name


## parallel info
parallel_num=100 ## number of input questions processed in parallel, if you have more model apis, you can set it larger.
batch_size=100000 ## number of input questions loaded for generation (you can set as the size of the dataset)

for n_samples in 40  ; do  ## to be noted, each input question will generate [ n_samples * model_num ] responses

## sampling params
max_tokens=2048
temperature=0.8
top_p=1

## MCTS hyper-parameters
tau=0.1 
alpha=0.01 
width=$(( model_num * n_samples / 3 ))
assert "[ $width -ge $model_num ]"


for topk_child in 1 ; do # we only keep the top-1 child 

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
                                     
