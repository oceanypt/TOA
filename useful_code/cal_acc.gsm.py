import sys
import json
import numpy as np
import re
import sys
import json
import numpy as np
sys.path.append('/mnt/2050data/haiye/ensemble_inference/ensemble_inference/code')
from flops import *
import os
from math_parser import *
import random
from tqdm import tqdm



models_of_flops = [
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/llama-3.1-70b-instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/Mixtral-8x22B-Instruct-v0.1.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/mistral-large-instruct-2407.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/Qwen2-72B-Instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/wizardlm-2-8x22b.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/llama-3.1-8b-instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/Qwen2-7B-Instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/Yi-1.5-9B-Chat-16K.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/Mistral-7B-Instruct-v0.2.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/Mistral-7B-Instruct-v0.3.json'
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
    if model_name not in model_flops_per_token:
        return 0.0
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
    keeped = {}
    with open(path, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            if item['id'] in keeped:
                continue
            id_to_item[item['id']] = item
            keeped[item['id']] = None
    return id_to_item


d1 = id_to_items(sys.argv[1])
num_sc = int(sys.argv[2])

result_pass_k = {}
info_flops = []
chosen = {}

num_eval = len(d1)

def cal_acc():
    num_correct = 0
    info_flops = []

    for k1, v1 in tqdm(d1.items()):
        ids_for_eval = list(range(len(v1['responses'])))
        #print(len(ids_for_eval))
        random.shuffle(ids_for_eval)
        ids_for_eval = ids_for_eval[:num_sc]
    
        responses = [ v1['responses'][id] for id in ids_for_eval  ]
        gold_answer = float(v1['gold_answer'])

        all_preds = {}
    
        for response in responses:
            try:
                pred_answer = response.split('\n')[-1].replace(',','')
                pred_answer = float(re.findall(r'\d+\.\d+|\d+', pred_answer)[0])

                if pred_answer not in all_preds:
                    all_preds[pred_answer] = 0
                all_preds[pred_answer] += 1 #reward  #1
            
            except Exception as e:
                all_preds['0'] = 1
    
        #print(all_preds)
        final_pred = max(all_preds, key=lambda k: all_preds[k])
        if final_pred == gold_answer:
            num_correct += 1
        
        temp = 0.
        actions = [ v1['actions'][id] for id in ids_for_eval  ]
        total_tokens = [ v1['total_tokens'][id] for id in ids_for_eval  ]
        for action, n_token in zip(actions, total_tokens):
            temp += get_flops(action, n_token)
        info_flops.append(temp)
    
    return num_correct / num_eval, np.mean(info_flops)

acc_result, flops_result = [], []
for _ in range(10):
    _acc, _flops = cal_acc()
    acc_result.append(_acc)
    flops_result.append(_flops)
    

print()
print(f'Num: {num_eval}')
print(f'n_samples: {num_sc}')
print(f'random trails: {len(acc_result)}')
print(f"Acc: { np.mean(acc_result) } ({np.std(acc_result)})")
print(f"Avg. flops per question: {'{:.2e}'.format(np.mean(flops_result))}")





