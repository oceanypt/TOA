from datasets import load_dataset
import sys
import json
import jsonlines
import random

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)


ds = load_dataset("princeton-nlp/llama3-ultrafeedback-armorm", cache_dir='./')


saved_id = {}
data_to_save = []
for i, item in enumerate(ds['train']):
    #print (item)
    if item['prompt_id'] in saved_id:
        continue
    new_item = {}
    new_item['id'] = item['prompt_id']
    new_item['instruction'] = item['prompt']

    new_item['dataset'] = 'ultrafeedback'

    data_to_save.append(new_item)
    
    
    saved_id[item['prompt']] = None

random.shuffle(data_to_save)

save_data_to_json(data_to_save, f'train.ultrafeedback.num={len(data_to_save)}.jsonl')
save_data_to_json(data_to_save[:10000], f'train.ultrafeedback.part_1.num={len(data_to_save[:10000])}.jsonl')
save_data_to_json(data_to_save[10000:20000], f'train.ultrafeedback.part_2.num={len(data_to_save[10000:20000])}.jsonl')
save_data_to_json(data_to_save[20000:30000], f'train.ultrafeedback.part_3.num={len(data_to_save[20000:30000])}.jsonl')
save_data_to_json(data_to_save[30000:40000], f'train.ultrafeedback.part_4.num={len(data_to_save[30000:40000])}.jsonl')
save_data_to_json(data_to_save[40000:], f'train.ultrafeedback.part_5.num={len(data_to_save[40000:])}.jsonl')




saved_id = {}
data_to_save = []
for i, item in enumerate(ds['test']):
    #print (item)
    if item['prompt_id'] in saved_id:
        continue
    new_item = {}
    new_item['id'] = item['prompt_id']
    new_item['instruction'] = item['prompt']

    new_item['dataset'] = 'ultrafeedback'

    data_to_save.append(new_item)
    
    saved_id[item['prompt']] = None
    

random.shuffle(data_to_save)

save_data_to_json(data_to_save, f'test.ultrafeedback.num={len(data_to_save)}.jsonl')


