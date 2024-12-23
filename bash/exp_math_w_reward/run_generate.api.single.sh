pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False

## mode
mode=ensemble_sample_N

## I / O params
task=math
data_name=math.test.num=100.jsonl 
input=../../data/$task/$data_name
mkdir ../../output/$task/
save_mode='a'


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_${1}/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
config_name=${1}.reward=ArmoRM
short_config_name=$config_name

## generation params
batch_size=1000  # the number of data for generation and save for one generation
parallel_num=500 # how many data points for generation for one time
save_mode='a'

for n_samples in $2 ; do

## sampling params
max_tokens=2048
temperature=0.7
top_p=1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl


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
                                     --batch_size $batch_size

done;
                                     
