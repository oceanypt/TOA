import sys
import json
import numpy as np
sys.path.append('../code/')
from flops import *
import os
from tqdm import tqdm

models_of_flops = [
    '../model_configs/llama-3.1-70b-instruct.json',
    '../model_configs/Mixtral-8x22B-Instruct-v0.1.json',
    '../model_configs/mistral-large-instruct-2407.json',
    '../model_configs/Qwen2-72B-Instruct.json',
    '../model_configs/wizardlm-2-8x22b.json',
    '../model_configs/llama-3.1-8b-instruct.json',
    '../model_configs/Qwen2-7B-Instruct.json',
    '../model_configs/Yi-1.5-9B-Chat-16K.json',
    '../model_configs/Mistral-7B-Instruct-v0.2.json',
    '../model_configs/Mistral-7B-Instruct-v0.3.json',
    '../model_configs/deepseek-math-7b-rl.json',
    '../model_configs/gemma-2-9b-it.json'
]
model_flops_per_token = dict()
model_names = []
c_ctl = 15000
for model_path in models_of_flops:
    with open(model_path, 'r') as f:
        config = json.load(f)
        #print (f"\n\n{config}\n\n")
        model_name = list(config['policy_model'].keys())[0]
        model_names.append(model_name)
        #print (f"\n\n{model_name}\n\n")
        config_path = os.path.join(config['policy_model'][model_name]['path_to_model'], "config.json")

        flops_per_token = cal_flops_per_token(config_path, c_ctl)
        model_flops_per_token[model_name] = flops_per_token  #"{:.2e}".format(flops_per_token)
print (model_flops_per_token)


def get_flops(model_name, token_n):
    for temp in model_names:
        if temp.lower() in model_name.lower():
            model_name = temp
            break
    return model_flops_per_token[model_name] * token_n




def id_to_score(path):
    id_to_score = {}
    with open(path, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            id_to_score[item['id']] = item['rewards']
    return id_to_score

def id_to_items(path):
    id_to_item = {}
    with open(path, 'r') as f:
        content = f.readlines()
        for i, item in enumerate(content):
            #print (i)
            item = json.loads(item)
            id_to_item[item['id']] = item
    return id_to_item


d1 = id_to_items(sys.argv[1])


max_rewards = []
median_rewards = []
avg_rewards = []
top3_rewards, top5_rewards, top10_rewards, top10_std_rewards = [], [], [], []
all_rewards = []
best_model = {}
info_flops = []

num_eval = 0

chosen = {}


for k1, v1 in tqdm(d1.items()):
        max_rewards.append(
           max(v1['rewards']   )
        )
        median_rewards.append(
            np.median(v1['rewards'])
        )
        avg_rewards.append(
            np.mean(v1['rewards'])
        )
        top3_rewards.append(
            np.mean(np.sort(v1['rewards'])[-3:] )
        )
        top5_rewards.append(
            np.mean(np.sort(v1['rewards'])[-5:]) 
        )
        top10_rewards.append(
            np.mean(np.sort(v1['rewards'])[-10:]) 
        )
        top10_std_rewards.append(
            np.std(np.sort(v1['rewards'])[-10:]) 
        )
        
        all_rewards += v1['rewards']
        num_eval += 1
        
        #print (v1['best_model'])
        
        if v1['best_model'] not in best_model:
            best_model[v1['best_model']] = 0
        best_model[v1['best_model']] += 1
        
        if 'actions' in v1:
            actions = v1['actions']
            for action in actions:
                if action not in chosen:
                    chosen[action] = 0
                chosen[action] += 1
        else:
            actions = v1['responses']
            for model_name in actions.keys():
                if model_name not in chosen:
                    chosen[model_name] = 0
                chosen[model_name] += len(actions[model_name])
        ## cal flops
        
        temp = 0.
        for action, n_token in zip(v1['actions'], v1['total_tokens']):
            temp += get_flops(action, n_token)
        info_flops.append(temp)
  

print(f'N_eval: {num_eval}')
print(f'Num: {len(max_rewards)}')
print(f'Max: {np.mean(max_rewards)}')
print(f'Top 3: {np.mean(top3_rewards)}')
print(f'Top 5: {np.mean(top5_rewards)}')
print(f'Top 10: {np.mean(top10_rewards)}')
print(f'Top 10 (std): {np.mean(top10_std_rewards)}')

#print(f'Med: {np.mean(median_rewards)}')
print(f'Avg: {np.mean(avg_rewards)}')
print(f'std: {np.std(all_rewards)}')
print('--------------------------')
print('Info. Flops')
print(f"Avg. flops per question: {'{:.2e}'.format(np.mean(info_flops))}") 
print(f"Total. flops: {'{:.2e}'.format(np.sum(info_flops))}") 
print('--------------------------')
print("Info. Best Model")
best_model = dict(sorted(best_model.items(), key=lambda x: x[1], reverse=True))
for k, v in best_model.items():
    print (f"{k}: {v}")
print('--------------------------')
print("Info. Actions")
chosen = dict(sorted(chosen.items(), key=lambda x: x[1], reverse=True))
for k, v in chosen.items():
    print (f"{k}: {v}")


print(f'\nINFO:')
print(f'Avg. rewards: {np.mean(all_rewards)}')

