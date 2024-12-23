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

from tqdm import tqdm

models_of_flops = [
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/llama-3.1-70b-instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/Mixtral-8x22B-Instruct-v0.1.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/mistral-large-instruct-2407.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/Qwen2-72B-Instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/wizardlm-2-8x22b.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/llama-3.1-8b-instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/Qwen2-7B-Instruct.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/Yi-1.5-9B-Chat-16K.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/Mistral-7B-Instruct-v0.2.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference/model_configs/Mistral-7B-Instruct-v0.3.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/deepseek-math-7b-rl.json',
    '/mnt/2050data/haiye/ensemble_inference/ensemble_inference//model_configs/gemma-2-9b-it.json'
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
    if n - c < k:
        #print (f"n - c < k")
        return 1.0
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
correct_ratio = []
num_correct = 0
n_samples = 0

for k1, v1 in tqdm(d1.items()):
    responses = v1['responses']
    rewards = v1['rewards']
    
    n_samples = len(v1['responses'])
    
    uni_response = {r: None for r in responses}
    #print(len(uni_response))
    
    gold_answer = v1['gold_answer']
    #print(gold_answer)
    
    n = len(responses)
    corrects = 0
    
    all_preds = {}
    
    for response, reward in zip(responses, rewards):
    #for response in uni_response.keys():
        try:
            pred_answer = extract_answer(response, "math", use_last_number=True)
            gold_answer = extract_gold_answer(gold_answer)

            if pred_answer not in all_preds:
                all_preds[pred_answer] = 0
            all_preds[pred_answer] += 1 #reward  #1
            
            if pred_answer == gold_answer:
                corrects += 1

        except Exception as e:
            pass
    #corrects = np.sum(np.array(v1['rewards']) == 0.8)
    

    #print (f"{n} - {corrects}")
    #for k in range(1, n+1, 2): #[ 1, 3, 5, 10 ]:
    #for k in [ 1, 3, 5, 10 ]:
    #for k in [ 1, 2, 3, 5, 10, 32, 64 ]:
    #for k in [64]:
    for k in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 32, 40, 128]:
        if k not in result_pass_k:
            result_pass_k[k] = []
        #print (f"n: {n} | corrects: {corrects}")
        score = pass_k(n, corrects, k)
        result_pass_k[k].append(score)  
    
    correct_ratio.append(corrects / n)
    
    final_pred = max(all_preds, key=lambda k: all_preds[k])
    if final_pred == gold_answer:
        num_correct += 1

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
print(f'n_samples: {n_samples}')

print("Info. Pass@K")
for k, v in result_pass_k.items():
    print (f"{k}: {np.mean(v)}")
print('--------------------------')
print(f"Correction Ration: {np.mean(correct_ratio)}")

print('--------------------------')
print(f"Acc: { num_correct / num_eval }")

print('--------------------------')
print("Info. Actions")
chosen = dict(sorted(chosen.items(), key=lambda x: x[1], reverse=True))
for k, v in chosen.items():
    print (f"{k}: {v}")
    
print('--------------------------')
print('Info. Flops')
print(f"Avg. flops per question: {'{:.2e}'.format(np.mean(info_flops))}") 
print(f"Total. flops: {'{:.2e}'.format(np.sum(info_flops))}") 



