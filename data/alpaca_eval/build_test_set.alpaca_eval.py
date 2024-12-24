import sys
import json
import jsonlines
import random

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)


data = json.load(open('alpaca_eval_gpt-4-turbo-2024-04-09.json', 'r'))
data_to_save = []
for i, item in enumerate(data):
    #{"instruction": "What is some cool music from the 1920s?", "dataset": "helpful_base", "id": 3, "profession": "Counselor", "preference_1": "I prefer the AI model to provide a diverse range of music genres from the 1920s, including jazz, blues, swing, and other popular styles of the era. It would be great if the AI model can recommend both well-known songs and lesser-known gems, highlighting the cultural and historical significance of each recommendation.", "preference_2": "I prefer the AI model to provide accurate and well-researched information in its responses. It should prioritize factual accuracy and reliability in its answers, ensuring that the inf
    new_item = {}
    new_item['instruction'] = item['instruction']
    new_item['dataset'] = item['dataset']
    new_item['id'] = i

    data_to_save.append(new_item)

save_data_to_json(data_to_save, f'alpaca_eval.num={len(data_to_save)}.jsonl')

random.shuffle(data_to_save)

save_data_to_json(data_to_save[:30], f'alpaca_eval.num=30.jsonl')
save_data_to_json(data_to_save[:100], f'alpaca_eval.num=100.jsonl')
save_data_to_json(data_to_save[:200], f'alpaca_eval.num=200.jsonl')





    

    
