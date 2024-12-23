import numpy as np
import sys
import json
import jsonlines
import numpy as np
import os



def save_data_to_json(strings, file_name, mode='w'):
    with open(file_name, mode, encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def cal_score(ref_file, hyp_file, device=[0]):
    refs = open(ref_file, 'r').readlines()
    hyps = open(hyp_file, 'r').readlines()
    
    refs, hyps = [ ref.strip() for ref in refs ], [ hyp.strip() for hyp in hyps ]
    
    assert len(refs) == len(hyps)
    
    data_to_save = [
                {
                    "hypothesis": hyp,
                    "reference": ref
                } for hyp, ref in zip(hyps, refs)
            ]
    
    save_data_to_json(data_to_save, "./cache_input.jsonl")
    
    script_path = f"./run_command.sh"
    import subprocess
    subprocess.run(["bash", script_path, 'cache_input.jsonl', 'cache_output.jsonl', str(device[0]) ])
    
    scores = []
    with open('./cache_output.jsonl', 'r') as f:
        data = f.readlines()
        for item in data:
            item = json.loads(item)
            scores.append(float(item['prediction']))

    return len(refs), np.mean(scores)
    
    

    # scores = []
    # with open('./cache_output.jsonl', 'r') as f:
    #     data = f.readlines()
    #     for item in data:
    #         item = json.loads(item)
    #         scores.append(float(item['prediction']))
    
    # with open(input_file, 'r') as f:
    #     data = f.readlines()
    #     data = [ json.loads(item) for item in data ]
    #     print (data[0])
    #     print ('\n\n------------------------------\n\n')
        
    #     data_to_save = [
    #             {
    #                 "hypothesis": item['best_response'],
    #                 "reference": item['en']
    #             } for item in data
    #         ]
    #     save_data_to_json(data_to_save, "./cache_input.jsonl")
    #     ## path to bash command
    #     script_path = f"/mnt/data/haiye/llms/metricx-23/metricx/run_command.sh"
    #     import subprocess
    #     subprocess.run(["bash", script_path, 'cache_input.jsonl', 'cache_output.jsonl', str(device[0]) ])
        
    #     scores = []
    #     with open('./cache_output.jsonl', 'r') as f:
    #         data = f.readlines()
    #         for item in data:
    #             item = json.loads(item)
    #             scores.append(float(item['prediction']))
        
    #     final_result = {
    #         "Found_data_num": len(data),
    #         "Metricx-23-XXL": f"{np.mean(scores)}" 
    #     }
    #     print (f"\n\n---> cur result: {np.mean(scores)}\n\n")
        
    # return final_result


# ALMA-13B-R/
# GPT-4-1106-preview/
# WMT_Winners/

# hyp_file = 'ALMA/outputs/wmt22_outputs/ALMA-13B-LoRA/csen/test.cs-en.en'
# ref_file = 'ALMA/outputs/wmt22_outputs/wmt-testset/csen/test.cs-en.en'

hyp_file = 'ALMA/outputs/wmt22_outputs/{model_name}/{lang}en/test.{lang}-en.en'
ref_file = 'ALMA/outputs/wmt22_outputs/wmt-testset/{lang}en/test.{lang}-en.en'

model_names = [ 'ALMA-13B-R', 'GPT-4-1106-preview', 'WMT_Winners' ]
langs = ['de', 'cs', 'is', 'zh', 'ru']

print()
print()
for model_name in model_names:
    for lang in langs:
        #print (f"--> {model_name}: {lang} to en")
        #print (ref_file.format(lang = lang))
        #print(hyp_file.format(model_name = model_name, lang = lang))
        num, score = cal_score(ref_file.format(lang = lang), hyp_file.format(model_name = model_name, lang = lang), device=[0])
        print (f"--> {model_name}: {lang} to en (num = {num})")
        print (f"--> score: {score}")
        print()
    print('****************************')      



