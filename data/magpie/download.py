from datasets import load_dataset
import json
import jsonlines


def save_data_to_json(strings, file_name, mode='w'):
    with open(file_name, mode, encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)
          
# ds = load_dataset("Magpie-Align/Magpie-Air-300K-Filtered", cache_dir = './')
# ds = load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", cache_dir = './')
# ds = load_dataset("Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered", cache_dir = './')
# ds = load_dataset("Magpie-Align/Magpie-Qwen-Air-300K-Filtered", cache_dir = './')

#print (ds['train'][0])

#for item in ds['train']:
#    assert len(item['conversations']) == 2

data_to_save = []
data_name = "Magpie-Qwen2-Pro-300K-Filtered" #Magpie-Align/Magpie-Qwen-Air-300K-Filtered
ds = load_dataset(f"Magpie-Align/{data_name}", cache_dir = './')
for item in ds['train']:
    conv = item['conversations']
    inst = conv[0]['value']
    output = conv[1]['value']
    
    new_item = {
        "id": item['uuid'],
        "instruction": inst,
        "ori_response": output,
        "dataset": data_name
    }
    
    data_to_save.append(new_item)

save_data_to_json(data_to_save, f"./{data_name}.jsonl")
    
    
