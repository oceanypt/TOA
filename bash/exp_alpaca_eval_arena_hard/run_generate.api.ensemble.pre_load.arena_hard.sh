pip install einops
pip install accelerate
export VLLM_USE_MODELSCOPE=False

## mode
mode=ensemble_sample_N

## I / O params
task=alpaca_eval_arena_hard
data_name=arena_hard_v0.1.num=500.jsonl  #alpaca_eval.num=805.part_1.num=100.jsonl #alpaca_eval.num=30.jsonl
input=../../data/$task/$data_name
mkdir ../../output/$task/

## model params
###--> reward=ArmoRM.json
# model_num=5
# root_configs=../launch_large_models_sglang/server_configs_num=5/
# path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=0.json
# config_name=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
# short_config_name=$config_name

# ###--> reward=ArmoRM.json
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_mistral-large-instruct-2407/
# path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
# config_name=Mis-large-2407.reward=ArmoRM
# short_config_name=$config_name

# ###--> reward=ArmoRM.json
# model_num=1
# root_configs=../launch_large_models_sglang/server_configs_lla-3.1-70b/
# path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
# config_name=lla-3.1-70b.reward=ArmoRM
# short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=5
root_configs=../launch_large_models_sglang/server_configs_Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
config_name=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_config_magpie_llama-3-8b-inst-train_data=1_8_80000/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=1.json
config_name=magpie_llama-3-8b-inst-train_data=1_8_80000.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_DPO/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_DPO_v2/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v2.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_DPO_v3/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_SimPO/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.SimPO.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_SimPO_v2/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.SimPO_v2.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_DPO_1p_3n/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b_DPO_1p_3n_v2/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n_v2.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=MagAir_part_1_to_8_MCTS.n=20.llama3-8b.DPO_1p_1n.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_llama-3-8b-instruct-simpo-v2/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=llama-3-8b-instruct-simpo-v2.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_SimPO/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_SimPO.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=128/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=128.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air-part_1_to_8.num=80000.MCTS.n_samples=20.1p_1n.llama3-8b-instruct_SimPO_beta=2.5/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.1p_1n.llama3-8b-instruct_SimPO_beta=2.5.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air-part_1_to_8.num=80000.MCTS.n_samples=20.1p_1n.llama3-8b-instruct_SimPO_lr=5e-7_beta=2.5/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.1p_1n.llama3-8b-instruct_SimPO_lr=5e-7_beta=2.5.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=64/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.1p_1n.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=64.reward=ArmoRM
short_config_name=$config_name



###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=16/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.1p_1n.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=16.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM
short_config_name=$config_name

###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=2
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name

###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_llama-3-8b-instruct-simpo-v2_MCTS.n_samples=16.pi=0_ni=20/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=llama-3-8b-instruct-simpo-v2_MCTS.n_samples=16.pi=0_ni=20.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name

###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=128/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=128.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=2e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=2e-7_batch=32.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=128/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=128.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback_magpie_air.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=128/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback_magpie_air.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=128.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=128/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=128.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=128/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=128.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=64/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=64.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name



###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name

###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=64/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=64.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name


###--> reward=ArmoRM.json
seed_id=1
model_num=1
root_configs=../launch_large_models_sglang/server_configs_Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32_beta=0.05/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32_beta=0.05.reward=ArmoRM.seed=${seed_id}
short_config_name=$config_name



###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_gemma-2-9b-it/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=7.json
config_name=gemma-2-9b-it.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=1
root_configs=../launch_large_models_sglang/server_configs_${2}/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=6.json
config_name=${2}.reward=ArmoRM
short_config_name=$config_name


###--> reward=ArmoRM.json
model_num=4
root_configs=../launch_large_models_sglang/server_configs_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407/
path_reward_config=/mnt/data/haiye/ensemble_inference/ensemble_inference/model_configs/reward_ArmoRM.gpus/reward=ArmoRM.gpu=3.json
config_name=Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM
short_config_name=$config_name


## generation params
batch_size=1000  # the number of data for generation and save for one generation
parallel_num=200 # how many data points for generation for one time
save_mode='a'

for n_samples in $1 ; do

## sampling params
max_tokens=2048
temperature=0.7 #0.001 #0.001 #0.7 #0.7
top_p=1

output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl
#output=../../output/$task/${data_name}.mode=${mode}.model_num=${model_num}.config=${short_config_name}.concise_v2.n_samples=${n_samples}.temp=${temperature}.top_p=${top_p}.jsonl


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
                                     
