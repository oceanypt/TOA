import pandas as pd
import re
import random
import jsonlines
import json

#template = "{}\n\nLet's think step by step. Output the final answer in the last line."
template = "{question}\n\nLet's think step by step. In the last new line, put the final answer in \"\\boxed{{}}\""

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def build_dataset(path):
    #df = pd.read_parquet(path)
    with open(path, 'r') as f:
        data = f.readlines()
    
    data_to_save = []
    for index, row in enumerate(data):
        row = json.loads(row)
        q = row['problem']
        a = row['solution']
        
        new_item = {}
        new_item['instruction'] = template.format(question = q) #q.strip() #template.format(q)
        new_item['id'] = row['unique_id']
        new_item['tag'] = 'math_500_openai'
        new_item['gold_response'] = a.strip()
        new_item['gold_answer'] = row['answer']
        
        data_to_save.append(new_item)
        
    return data_to_save



data = build_dataset("./math_test_openai_500.jsonl")

random.shuffle(data)
save_data_to_json(data[:100], f'./math.test.num=100.jsonl')
save_data_to_json(data[:200], f'./math.test.num=200.jsonl')
save_data_to_json(data, f'./math.test.num={len(data)}.jsonl')






    
