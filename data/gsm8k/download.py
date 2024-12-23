import pandas as pd
import re
import random
import jsonlines

#template = "{}\n\nLet's think step by step. Output the final answer in the last line."
template = "{}\n\nLet's think step by step. In the last new line, output the final answer in the format: Final answer: {{answer value}}."

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def build_dataset(path):
    df = pd.read_parquet(path)
    data_to_save = []
    for index, row in df.iterrows():
        q = row['question']
        a = row['answer']
        
        new_item = {}
        new_item['instruction'] = template.format(q) #q.strip() #template.format(q)
        new_item['id'] = index
        new_item['tag'] = 'gsm8k'
        new_item['gold_response'] = a.strip()
        text = a.split('###')[1].replace(',','')
        new_item['gold_answer'] = re.findall(r'\d+\.\d+|\d+', text)[0]
        
        data_to_save.append(new_item)
        
    return data_to_save


splits = {'train': 'socratic/train-00000-of-00001.parquet', 'test': 'socratic/test-00000-of-00001.parquet'}
#df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])

data = build_dataset("hf://datasets/openai/gsm8k/" + splits["train"])
save_data_to_json(data, f'./gsm8k.train.num={len(data)}.jsonl')

data = build_dataset("hf://datasets/openai/gsm8k/" + splits["test"])

random.shuffle(data)
save_data_to_json(data[:100], f'./gsm_w_template/gsm8k.test.num=100.jsonl')
save_data_to_json(data[:200], f'./gsm_w_template/gsm8k.test.num=200.jsonl')
save_data_to_json(data, f'./gsm_w_template/gsm8k.test.num={len(data)}.jsonl')






    
