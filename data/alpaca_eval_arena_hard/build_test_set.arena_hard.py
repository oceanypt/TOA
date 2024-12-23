import sys
import json
import jsonlines
import random

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)


with open('arena_hard_v0.1_question.jsonl', 'r') as f:
    data = f.readlines()
    data = [ json.loads(item) for item in data ]

data_to_save = []
for item in data:
    #{"instruction": "What is some cool music from the 1920s?", "dataset": "helpful_base", "id": 3, "profession": "Counselor", "preference_1": "I prefer the AI model to provide a diverse range of music genres from the 1920s, including jazz, blues, swing, and other popular styles of the era. It would be great if the AI model can recommend both well-known songs and lesser-known gems, highlighting the cultural and historical significance of each recommendation.", "preference_2": "I prefer the AI model to provide accurate and well-researched information in its responses. It should prioritize factual accuracy and reliability in its answers, ensuring that the inf
    new_item = {}
    new_item['instruction'] = item['turns'][0]['content']
    new_item['id'] = item['question_id']
    new_item['category'] = item['category']
    new_item['cluster'] = item['cluster']
    

    data_to_save.append(new_item)

save_data_to_json(data_to_save, f'arena_hard_v0.1.num={len(data_to_save)}.jsonl')

random.shuffle(data_to_save)

save_data_to_json(data_to_save[:30], f'arena_hard_v0.1.num=30.jsonl')
save_data_to_json(data_to_save[:100], f'arena_hard_v0.1.num=100.jsonl')
save_data_to_json(data_to_save[:200], f'arena_hard_v0.1.num=200.jsonl')
