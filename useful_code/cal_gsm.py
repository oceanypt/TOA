import sys
import json
import numpy as np
import re

def sigmoid(x, tau=5):
    return 1 / (1 + np.exp(-x / tau))

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


max_rewards = []
median_rewards = []
avg_rewards = []
topk_rewards = []
all_rewards = []
best_model = {}

num_eval = 0

chosen = {}

correct, test_num = 0, 0

for k1, v1 in d1.items():
        max_rewards.append(
           max(v1['rewards']   )
        )
        median_rewards.append(
            np.median(v1['rewards'])
        )
        avg_rewards.append(
            np.mean(v1['rewards'])
        )
        topk_rewards.append(
            np.mean(np.sort(v1['rewards'])[-3:] )
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
        
        
        try:
            gold_answer = float(v1['gold_answer'])
            #print (v1['best_response'])
            pred_answer = v1['best_response'].split('\n')[-1].replace(',','')
            #print (pred_answer)
            pred_answer = float(re.findall(r'\d+\.\d+|\d+', pred_answer)[0])
            if gold_answer == pred_answer:
                correct += 1
            test_num += 1
        except Exception as e:
            print (f"{e}")
        

print(f'N_eval: {num_eval}')
print(f'Num: {len(max_rewards)}')
print(f'Acc: {correct} / {test_num} = {correct / test_num:.4f}')
print(f'Max: {np.mean(max_rewards)}')
print(f'Top 3: {np.mean(topk_rewards)}')
#print(f'Med: {np.mean(median_rewards)}')
print(f'Avg: {np.mean(avg_rewards)}')
print(f'std: {np.std(all_rewards)}')
print('--------------------------')
print("Info. Best Model")
for k, v in best_model.items():
    print (f"{k}: {v}")
print('--------------------------')
print("Info. Actions")
for k, v in chosen.items():
    print (f"{k}: {v}")


print(f'\nINFO:')
print(f'Avg. rewards: {np.mean(all_rewards)}')

