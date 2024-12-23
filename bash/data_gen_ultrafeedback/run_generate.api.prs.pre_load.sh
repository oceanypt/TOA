pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False

## mode
mode=PRS

## I / O params
task=ultra_feedback
train_part_id=$1
data_name=train.ultrafeedback.num=59876.split=${train_part_id}.jsonl
input=../../data/${task}/splits/$data_name
mkdir ../../output/$task/splits/
save_mode='a'


# ## model params
# ###--> reward=ArmoRM.json
# model_num=1
# root_configs=./server_configs_${2}/
# path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
# config_name=${2}.reward=ArmoRM
# short_config_name=$config_name

# ## model params
# ###--> reward=ArmoRM.json
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_qwen2-7b/
# path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=$((train_part_id % 8)).json
# config_name=qwen2-7b.reward=ArmoRM
# short_config_name=$config_name


## model params
###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_lla3.1-8b/
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=$((train_part_id % 8)).json
config_name=lla3.1-8b.reward=ArmoRM
short_config_name=$config_name


## generation params
batch_size=10000  # the number of data for generation and save for one generation
parallel_num=${3} # how many data points for generation for one time
save_mode='a'

for n_samples in $2 ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

#path_to_refine_template=../../prompts/refinement_wo_feedback.inst_following.v2.txt
path_to_refine_template=../../prompts/refinement_wo_feedback.inst_following.v13.txt

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.temp_v=13.jsonl


#CUDA_VISIBLE_DEVICES=$GPU
python ../../code/ensemble_inference.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --path_reward_config $path_reward_config \
                                     --path_to_refine_template $path_to_refine_template \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size

done;
                                     
