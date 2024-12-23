
# export IS_ALPACA_EVAL_2=False
# #path1=Mistral_7b_v2_N=32/mistral_7b_v2_prand.json
# #path1=outputs/alpaca_eval_set.w_preference_by_gpt-3.5.part_1.num=200.prefer=common.generator=train_part-1.tree_search.N=16.temp=0.7.top_p=0.9.json
# path1=outputs/alpaca_eval_set.w_preference_by_gpt-3.5.part_1.num=200.prefer=common.generator=train_part-1.sample_N.N=16.temp=0.7.top_p=0.9.json
# path2=outputs/gpt-4-turbo-2024-04-09.json




# ## xiaoyao's key
# export OPENAI_API_KEY=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

# ## skywork
# export OPENAI_API_BASE=https://gpt.singularity-ai.com/gpt-proxy
# export OPENAI_API_KEY=7f44815efadb67318d48b1ed6dd94123

# ## ensemble
# path=/mnt/data/haiye/ensemble_inference/ensemble_inference/output/alpaca_eval_arena_hard/alpaca_format/alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=4.config=models=lla3.1-8-ins_qwen2-7-ins_Yi1.5-9-16k-chat.mis-7-ins-v0.2.reward=ArmoRM.n_samples=16.json

# ## MCTS
# # path=/mnt/data/haiye/ensemble_inference/ensemble_inference/output/alpaca_eval_arena_hard/alpaca_format/alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=models=lla3.1-8-ins_qwen2-7-ins_Yi1.5-9-16k-chat.mis-7-ins-v0.2.reward=ArmoRM.n_samples=16.tau=0.1.alpha=0.01.width=21.topk_child=1.json

# ## baseline output
# reference_output=/mnt/data/haiye/ensemble_inference/ensemble_inference/data/alpaca_eval_arena_hard/gpt4_1106_preview.json

# alpaca_eval --model_outputs $path  --reference_outputs $reference_output





## xiaoyao's key
#export OPENAI_API_KEY=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

## skywork
export OPENAI_API_BASE=https://gpt.singularity-ai.com/gpt-proxy
export OPENAI_API_KEY=7f44815efadb67318d48b1ed6dd94123

# ## yehai
# export OPENAI_API_BASE=https://api.chatanywhere.tech/v1
# export OPENAI_API_KEY=sk-L0dfb9PvTfvPFwInt2UAaubaIcR8vkvaS9Yr5LlgTDiZKko9



root_output=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/output/alpaca_eval_arena_hard/alpaca_format/

## ensemble
#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Qwen2-72B-Instruct.reward=ArmoRM.n_samples=32


## MCTS
#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=config.models=Mixtral-8x22B-Instruct-v0.1_Qwen2-72B-Instruct_llama-3.1-70b-instruct_Yi-1.5-34b-chat-16k.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=config.models=Mixtral-8x22B-Instruct-v0.1_Qwen2-72B-Instruct_llama-3.1-70b-instruct_Yi-1.5-34b-chat-16k.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=config.models=Mixtral-8x22B-Instruct-v0.1_Qwen2-72B-Instruct_llama-3.1-70b-instruct_Qwen1.5-110B-Chat.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=config.models=Mixtral-8x22B-Instruct-v0.1_Qwen2-72B-Instruct_llama-3.1-70b-instruct_Yi-1.5-34b-chat-16k.reward=ArmoRM.n_samples=64.tau=0.1.alpha=0.01.width=85.topk_child=1
#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=config.models=Mixtral-8x22B-Instruct-v0.1_Qwen2-72B-Instruct_llama-3.1-70b-instruct_Yi-1.5-34b-chat-16k.reward=ArmoRM.n_samples=64.tau=0.1.alpha=0.01.width=85.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=llama3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=InternRM.n_samples=32.tau=1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gpt4_1106_preview.n_samples=32.tau=1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gpt4_1106_preview.n_samples=64.tau=1.alpha=0.01.width=85.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gpt4_1106_preview.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1
#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gpt4_1106_preview.n_samples=64.tau=0.1.alpha=0.01.width=85.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gpt-4o-2024-05-13.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=Qwen1.5-110B-Chat.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=Qwen1.5-110B-Chat.n_samples=256.tau=0.1.alpha=0.01.width=341.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gpt-4-turbo-2024-04-09.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=la3-8b_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM.initial=Qwen1.5-110B-Chat_refined_32.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=Together-MoA.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gemma-2-9b-it-WPO-HB.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1
#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=gemma-2-9b-it-WPO-HB.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b-inst_yi1.5-9b-chat_mistral-7b-inst-v0.2.reward=ArmoRM.initial=Together-MoA.n_samples=128.tau=0.1.alpha=0.01.width=170.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM_ultrafb_overall.initial=Together-MoA.n_samples=32.tau=1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_lla-3.1-70b_Yi-34b.reward=ArmoRM.initial=Together-MoA.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b-inst_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM_InternRM.initial=Together-MoA.n_samples=32.tau=0.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_lla-3.1-70b_Yi-34b.reward=ArmoRM.initial=1st_same_tree.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1


#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM.initial=Qwen1.5-110B-Chat.template_v4.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM.initial=Qwen1.5-110B-Chat.template_v5.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM.initial=Qwen1.5-110B-Chat.template_v5.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3-8b_qwen2-7b_yi1.5-9b_mistral-7b-v0.2.reward=ArmoRM.initial=Together-MoA.template_v4.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B-Inst-v0.1_Qwen2-72B-Inst_llama-3.1-70b-inst_Yi1.5-34B-16k-Chat.reward=ArmoRM.template_v4.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B-Inst-v0.1_Qwen2-72B-Inst_llama-3.1-70b-inst_Yi1.5-34B-16k-Chat.reward=ArmoRM.template_v6.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Yi1.5-34B-16k.reward=ArmoRM.template_v6.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1


#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wizard-8x22B.reward=ArmoRM.template_v6.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wizard-8x22B.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wizard-8x22B.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=64.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wizard-8x22B.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=32.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_lla-3.1-70b_Wizard-8x22B.Qwen1.5-110b.reward=ArmoRM.template_v7.n_samples=26.tau=0.1.alpha=0.01.width=43.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wizard-8x22B.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wizard-8x22B.reward=ArmoRM.template_v8.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1


# data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_dbrx.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_dbrx.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1f

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_dbrx.reward=ArmoRM.template_v7.n_samples=64.tau=0.1.alpha=0.01.width=106.topk_child=1.temp=0.7.top_p=1

# data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Qwen1.5-110b.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=6.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Qwen1.5-110b_dbrx.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=64.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Yi1.5-34b-16k.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v6.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v4.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v5.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1


#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v7.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v6.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v9.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v3.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v3.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.4.top_p=1


data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v3.n_samples=32.tau=0.1.alpha=0.01.width=80.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v3.n_samples=32.tau=0.1.alpha=0.01.width=40.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v10.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v11.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v11.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v11.n_samples=50.tau=0.1.alpha=0.01.width=83.topk_child=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v11.n_samples=16.tau=0.1.alpha=0.01.width=26.topk_child=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.part_6.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=6.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407_dsv20628.ArmoRM.temp_v11.n_samples=32.tau=0.1.alpha=0.01.width=64.topk_child=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=6.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407_dsv20628.ArmoRM.temp_v11.n_samples=32.tau=0.1.alpha=0.01.width=64.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v11.n_samples=16.tau=0.1.alpha=0.01.width=26.topk_child=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v12.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1


#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v13.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=6.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.Qw1.5-110b.reward=ArmoRM.temp_v13.n_samples=32.tau=0.1.alpha=0.01.width=64.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B.reward=ArmoRM.temp_v14.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=6.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.Qw1.5-110b.reward=ArmoRM.temp_v14.n_samples=32.tau=0.1.alpha=0.01.width=64.topk_child=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v14.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=PRS.model_num=1.config=mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=ensemble_sample_N.model_num=1.config=Mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=PRS.model_num=1.config=mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=15

data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=PRS.model_num=1.config=mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.part_1.num=100.jsonl.mode=PRS.model_num=1.config=lla-3.1-70b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.part_1.num=100.jsonl.mode=ensemble_sample_N.model_num=1.config=lla-3.1-70b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=ensemble_MoA.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.n_iter=4.num_agg=5.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_1.num=100.jsonl.mode=ensemble_sample_N.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.part_0.num=100.jsonl.mode=ensemble_sample_N.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_MoA.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.n_iter=4.num_agg=5.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1.temp_v=12

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=lla-3.1-70b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1.temp_v=13


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=magpie_llama-3-8b-train_data=1_8_80000.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=magpie_llama-3-8b-inst-train_data=1_8_80000.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=64.tau=0.1.alpha=0.01.width=85.topk_child=1.temp=0.7.top_p=1.temp_v=13


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=64.tau=0.1.alpha=0.01.width=106.topk_child=1.temp=0.7.top_p=1.temp_v=13

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v2.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM.n_samples=1.temp=0.3.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM.concise.n_samples=1.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM.concise_v2.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_v3.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=Llama-3-Instruct-8B-SimPO

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.SimPO.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=vanilla_dpo_300

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n.reward=ArmoRM.concise_v1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n.reward=ArmoRM.concise_v2.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagPro_p_1_to_6_MCTS.n=64.top=3.llama3-8b.DPO_1p_3n_v2.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagAir_part_1_to_8_MCTS.n=20.llama3-8b.DPO_1p_1n.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagAir_part_1_to_8_MCTS.n=20.llama3-8b.DPO_1p_1n.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=MagAir_part_1_to_8_MCTS.n=20.llama3-8b.DPO_1p_1n.reward=ArmoRM.n_samples=1.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=llama-3-8b-instruct-simpo-v2.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_SimPO.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=128.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=128.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=128.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.1p_1n.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=64.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.1p_1n.llama3-8b-instruct_DPO_1p_1n_lr=5e-7_batch=16.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.seed=2.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.3.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=2e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=20_lr=2e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Magpie-Air.part_1_to_8_MCTS.n_samples=20.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=16.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=128.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=64.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=64.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.7.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.3.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32.reward=ArmoRM.seed=1.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=64.reward=ArmoRM.seed=1.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32_beta=0.05.reward=ArmoRM.seed=1.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=80_lr=5e-7_batch=32_beta=0.05.reward=ArmoRM.seed=1.n_samples=1.temp=0.1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1.temp_v=2

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1.temp_v=2

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1.temp_v=2

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

data_name=Meta-Llama-3-8B-Instruct

data_name=Mistral-7B-Instruct-v0.2

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=16.temp=1.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=16.temp=1.top_p=1.temp_v=2

#data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1

#data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=16.temp=1.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=16.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=16.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=32.temp=1.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=32.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=32.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=32.temp=1.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=16.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=16.temp=1.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=32.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=gemma-2-9b-it.reward=ArmoRM.n_samples=32.temp=1.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=mis-7b-v0.2.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1.temp_v=2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3-8b.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=1.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=40_lr=5e-7_batch=128.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1


data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=20_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=20.large_models.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=lla-3.1-70b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=lla-3.1-70b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=PRS.model_num=1.config=Mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_Seq.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.llama3-8b-instruct.sft.data=MCTS_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_sft_data=MoA_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_sft_data=ensemble_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_sft_data=single_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_sft_data=MCTS_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_sft_data=MoA_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_sft_data=ensemble_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_sft_data=single_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=ultra_dpo_1004

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=16.tau=0.1.alpha=0.01.width=21.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=16.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=42.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=32.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=64.tau=0.1.alpha=0.01.width=85.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=96.tau=0.1.alpha=0.01.width=128.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=128.tau=0.1.alpha=0.01.width=170.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=64.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=160.tau=0.1.alpha=0.01.width=213.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N_MCTS.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=192.tau=0.1.alpha=0.01.width=256.topk_child=1.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=192.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=128.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=qwen2-7b.reward=ArmoRM.n_samples=768.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=qwen2-7b.reward=ArmoRM.n_samples=768.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=qwen2-7b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=qwen2-7b.reward=ArmoRM.n_samples=384.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=96.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_MoA.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=160.n_iter=4.num_agg=4.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=qwen2-7b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=768.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=768.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=128.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=256.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=384.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=512.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=PRS.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=640.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=192.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=192.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=128.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=96.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_Seq.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1.temp_v=13

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=128.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=256.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=384.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=512.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3.1-8b.reward=ArmoRM.n_samples=640.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=16.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=32.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=64.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=96.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=128.temp=0.7.top_p=1

data_name=alpaca_eval.num=200.jsonl.mode=ensemble_sample_N.model_num=4.config=lla3.1-8b_mis-7b-v0.2_yi-1.5-16k_qwen2-7b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=lla3-8b-instruct.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=qwen2-7b-instruct.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=ultra_dpo_2

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_sft_data=single_lla3.1_8b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_sft_data=prs_lla3.1_8b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_sft_data=prs_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30=prs_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30=prs_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30=prs_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30=prs_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.top_p=0.5

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30=prs_qwn2_7b.reward=ArmoRM.n_samples=1.temp=0.top_p=0.1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.ab=eos_token.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=Ultrafeedback.MCTS.n_samples=40.llama3-8b-instruct_DPO_pi=0_ni=30_lr=5e-7_batch=32.reward=ArmoRM.n_samples=1.temp=0.001.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30.reward=ArmoRM.n_samples=1.temp=0.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_dpo_data=small_scale_mcts_n=40.pi=0_ni=30.reward=ArmoRM.n_samples=1.temp=0.top_p=0.5

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=yi-1.5-16k.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_llama3-8b-instruct_sft_data=seq_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N.model_num=1.config=ultrafeedback_Qwen2-7B-Instruct_sft_data=seq_small.reward=ArmoRM.n_samples=1.temp=0.7.top_p=1

data_name=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qw2-72B_lla-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1.temp_v=13

path=$root_output/$data_name.json

#data_name=Together-MoA
#path=../../data/alpaca_eval_arena_hard/Together-MoA.json

## baseline output
reference_output=/mnt/2050data/haiye/ensemble_inference/ensemble_inference/data/alpaca_eval_arena_hard/gpt4_1106_preview.json
#reference_output=../../data/alpaca_eval_arena_hard/Together-MoA.json
#reference_output=../../data/alpaca_eval_arena_hard/Together-MoA-Lite.json
#base=alpaca_eval.num=805.jsonl.mode=ensemble_sample_N_MCTS.model_num=5.config=Mix-8x22B_Qwen2-72B_llama-3.1-70b_Wiza-8x22B_Mis-large-2407.reward=ArmoRM.template_v14.n_samples=32.tau=0.1.alpha=0.01.width=53.topk_child=1.temp=0.7.top_p=1
#base=alpaca_eval.num=805.part_0.num=100.jsonl.mode=ensemble_sample_N.model_num=1.config=Mis-large-2407.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1
#base=alpaca_eval.num=805.part_0.num=100.jsonl.mode=ensemble_sample_N.model_num=1.config=lla-3.1-70b.reward=ArmoRM.n_samples=160.temp=0.7.top_p=1
#reference_output=$root_output/$base.json




export HF_ENDPOINT=https://hf-mirror.com
alpaca_eval --model_outputs $path  --reference_outputs $reference_output

echo
echo "--> Reference:${reference_output}"

cd $root_output/
rm -rf $data_name
mv weighted_alpaca_eval_gpt4_turbo $data_name
realpath $data_name

#python mv.py weighted_alpaca_eval_gpt4_turbo  $data_name

# rm -rf $root_output/weighted_alpaca_eval_gpt4_turbo_$data_name
# mv $root_output/weighted_alpaca_eval_gpt4_turbo $root_output/weighted_alpaca_eval_gpt4_turbo_$data_name

