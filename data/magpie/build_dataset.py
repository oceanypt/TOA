import json
import jsonlines
import random



def save_data_to_json(strings, file_name, mode='w'):
    with open(file_name, mode, encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)
            
data_name = 'Magpie-Qwen-Air-300K-Filtered' #'Magpie-Qwen2-Pro-300K-Filtered' #'Magpie-Pro-300K-Filtered' #'Magpie-Air-300K-Filtered'

with open(f"./{data_name}.jsonl", 'r') as f:
    data = f.readlines()
    data = [ json.loads(item) for item in data ]
random.shuffle(data)

## 100
save_data_to_json(data[:100], f"./{data_name}/{data_name}.dev.num=100.jsonl")

## 200
save_data_to_json(data[:200], f"./{data_name}/{data_name}.dev.num=200.jsonl")

## 500
save_data_to_json(data[:500], f"./{data_name}/{data_name}.dev.num=500.jsonl")

## 1000
save_data_to_json(data[500:1500], f"./{data_name}/{data_name}.test.num=1000.jsonl")

data = data[1500:]

gap = 10000
for i in range(int(len(data) / gap)+1):
    save_data_to_json(data[i*gap : (i+1)*gap ], f"./{data_name}/{data_name}.train.part_{i+1}.num={len(data[i*gap : (i+1)*gap ])}.jsonl")

