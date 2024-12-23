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

template = "Translate the following sentence into English. Directly output the translated sentence: \n\n\"{sentence}\""

source='ru'

ds = load_dataset("haoranxu/WMT22-Test", f"{source}-en")
label = f"WMT22-Test_{source}_en"

#print (ds['test'][0])

## test set
test_items = []
for i, item in enumerate(ds['test']):
    print (item)
    new_item = {}
    new_item['id'] = i
    
    en = item[f"{source}-en"]['en']
    source_lang = item[f"{source}-en"][source]
    
    new_item['instruction'] = template.format(sentence = source_lang)
    new_item['en'] = en
    new_item[source] = source_lang
    new_item['dataset'] = label
    
    test_items.append(new_item)
    

save_data_to_json(test_items, f'test.{source}_to_en.num={len(test_items)}.jsonl')

random.shuffle(test_items)
save_data_to_json(test_items[:500], f'test.{source}_to_en.num={len(test_items[:500])}.jsonl')
save_data_to_json(test_items[:200], f'test.{source}_to_en.num={len(test_items[:200])}.jsonl')
save_data_to_json(test_items[:100], f'test.{source}_to_en.num={len(test_items[:100])}.jsonl')
save_data_to_json(test_items[:300], f'test.{source}_to_en.num={len(test_items[:300])}.jsonl')

