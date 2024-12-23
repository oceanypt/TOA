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

def get_answer(text):
    try:
        pred_answer = text.split('\n')[-1].replace(',','')
    
        pred_answer = float(re.findall(r'\d+\.\d+|\d+', pred_answer)[0])
    except:
        pred_answer = "#"
    
    return pred_answer

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
            pred_freq = {}
            pred_rewards = {}
            if isinstance(v1['responses'], list):
                responses = v1['responses']
            else:
                responses = []
                for a, b in v1['responses'].items():
                    responses += b
                    
            for response, c_reward in zip(responses, v1['rewards']):
                c_pred = get_answer(response)
                if c_pred not in pred_freq:
                    pred_freq[c_pred] = 0
                    pred_rewards[c_pred] = []
                pred_freq[c_pred] += 1
                pred_rewards[c_pred].append(c_reward)
            
            pred_rewards = {k: f'{np.mean(v):.3}' for k, v in pred_rewards.items()}
            #print (pred_rewards)
            pred_freq = { k: [v, pred_rewards[k]]  for k, v in pred_freq.items() }
            
            
            
            
            #filtered_dict = {k: v for k, v in pred_freq.items() if k != '#'}
            
            filtered_dict = {k: float(v[0]) for k, v in pred_freq.items() if k != '#'}

            pred_answer = max(filtered_dict, key=filtered_dict.get)
            
#             if gold_answer != pred_answer:
#                 print (f'\n\n')
#                 print (f'gold label: {gold_answer}')
#                 print (f'pred answer: {pred_answer}')
#                 print (f'pred label: {pred_freq}')
#                 print ('---> wrong')
            
            if gold_answer == pred_answer:
                correct += 1
            
            # if gold_answer == pred_answer:
            #     print ('---> correct')
            # else:
            #     print ('---> wrong')
            
            
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

