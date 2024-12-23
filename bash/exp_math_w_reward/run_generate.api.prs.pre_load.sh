pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False

## mode
mode=PRS

## I / O params
task=math
data_name=math.test.num=100.jsonl 
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'

###--> reward=ArmoRM.json
model_num=1
root_configs=./server_configs_${1} #../launch_large_models_sglang/server_configs_${1}/
config_name=${1}.reward=GSM
short_config_name=$config_name


###--> reward=Shepherd.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_${1}  #qwen2-7b  lla3.1-8b mis-7b-v0.2 yi-1.5-16k
path_reward_config=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_Qwen25_Math_RM.gpus/reward=Qwen25_Math_RM.gpu=0_1.json
config_name=${1}.reward=Qwen25_Math_RM
short_config_name=$config_name

## generation params
batch_size=1000000  # the number of data for generation and save for one generation
parallel_num=25 #250 # how many data points for generation for one time
save_mode='a'

for n_samples in $2 ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

path_to_refine_template=../../prompts/refinement_wo_feedback.math.v1.txt

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.temp_v=math_v1.jsonl


python ../../code/ensemble_inference.server_pre_load.fast.py --mode $mode \
                                     --input $input \
                                     --output $output \
                                     --root_configs $root_configs \
                                     --path_to_refine_template $path_to_refine_template \
                                     --path_reward_config $path_reward_config \
                                     --n_samples $n_samples \
                                     --max_tokens $max_tokens \
                                     --temperature $temperature \
                                     --top_p $top_p \
                                     --parallel_num $parallel_num \
                                     --save_mode $save_mode \
                                     --batch_size $batch_size



done;
                                     
