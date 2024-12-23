import sys
import json
import numpy as np
import re
import sys
import json
import numpy as np
sys.path.append('../../code/')
from flops import *
import os

models_of_flops = [
    '../../model_configs/llama-3.1-70b-instruct.json',
    '../../model_configs/Mixtral-8x22B-Instruct-v0.1.json',
    '../../model_configs/mistral-large-instruct-2407.json',
    '../../model_configs/Qwen2-72B-Instruct.json',
    '../../model_configs/wizardlm-2-8x22b.json',
    '../../model_configs/llama-3.1-8b-instruct.json',
    '../../model_configs/Qwen2-7B-Instruct.json',
    '../../model_configs/Yi-1.5-9B-Chat-16K.json',
    '../../model_configs/Mistral-7B-Instruct-v0.2.json',
    '../../model_configs/Mistral-7B-Instruct-v0.3.json'
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

def pass_k(n, c, k=1):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

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
        for item in content:
            item = json.loads(item)
            id_to_item[item['id']] = item
    return id_to_item


d1 = id_to_items(sys.argv[1])

result_pass_k = {}
info_flops = []
chosen = {}

num_eval = len(d1)


for k1, v1 in d1.items():
    responses = v1['responses']
    gold_answer = float(v1['gold_answer'])
    
    n = len(responses)
    corrects = 0
    
    for response in responses:
        try:
            pred_answer = response.split('\n')[-1].replace(',','')
            pred_answer = float(re.findall(r'\d+\.\d+|\d+', pred_answer)[0])
        
            if pred_answer == gold_answer:
                corrects += 1

        except Exception as e:
            pass
            #print(f"{e}")
    #for k in range(1, n+1, 2): #[ 1, 3, 5, 10 ]:
    #for k in [ 1, 3, 5, 10 ]:
    for k in [ 1, 2, 3, 5, 10 ]:
        if k not in result_pass_k:
            result_pass_k[k] = []
        score = pass_k(n, corrects, k)
        #print(score)
        result_pass_k[k].append(score)  
    
    temp = 0.
    for action, n_token in zip(v1['actions'], v1['total_tokens']):
        temp += get_flops(action, n_token)
    info_flops.append(temp)
    
    if 'actions' in v1:
        actions = v1['actions']
        for action in actions:
            if action not in chosen:
                chosen[action] = 0
            chosen[action] += 1

print()
print(f'Num: {num_eval}')
print("Info. Pass@K")
for k, v in result_pass_k.items():
    print (f"{k}: {np.mean(v)}")
print('--------------------------')

print("Info. Actions")
chosen = dict(sorted(chosen.items(), key=lambda x: x[1], reverse=True))
for k, v in chosen.items():
    print (f"{k}: {v}")
    
print('--------------------------')
print('Info. Flops')
print(f"Avg. flops per question: {'{:.2e}'.format(np.mean(info_flops))}") 
print(f"Total. flops: {'{:.2e}'.format(np.sum(info_flops))}") 



