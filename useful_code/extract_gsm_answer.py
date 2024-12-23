import argparse
import jsonlines
from tqdm import tqdm
import torch
import json
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List
import openai
import threading
from multiprocessing import Pool
import numpy as np
from reward import UltraRM, ArmoRM

import subprocess
from openai import OpenAI

def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def build_client(
  path_to_model,
  path_to_chat_template,
  api_key,
  gpu,
  port=8000  
):
    ## load policy models
    #gpu = '0'  # GPU编号
    #path_to_model = '/mnt/data/haiye/llms/Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa'  # 模型路径
    #api_key='token-abc123'
    #path_to_chat_template='/mnt/data/haiye/ensemble_inference/ensemble_inference/chat_templates/llama-3-instruct.jinja'
    # 构建命令
    try:
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key=api_key,
        )
        client.chat.completions.create(
                model=path_to_model,
                messages=[{"role": "user", "content": "Hi!" }],
                max_tokens=20,
                temperature=0.7,
                top_p=0.9,
                stop="<|eot_id|>",
                n=1,
            )
        return client, None
    except Exception as e:
        print(f"{e}")
    
    
    gpu_num = len(gpu.split(','))
    
    command = f"CUDA_VISIBLE_DEVICES={gpu} python -m vllm.entrypoints.openai.api_server " \
          f"--model {path_to_model} --dtype auto " \
          f"--api-key {api_key} " \
          f"--port {port} --chat-template {path_to_chat_template} --disable-log-stats --tensor-parallel-size {gpu_num}"
    process = subprocess.Popen(command, shell=True) # keep process to kill in the end
    #subprocess.Popen(command, shell=True)

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key=api_key,
    )
    while True:
        try:
            client.chat.completions.create(
                model=path_to_model,
                messages=[{"role": "user", "content": "Hi!" }],
                max_tokens=20,
                temperature=0.7,
                top_p=0.9,
                stop="<|eot_id|>",
                n=1,
            )
            break
        except Exception as e:
            print(f"{e}")
    
    return client, process
    

def load_models(config):
    ## load policy models
    policy_model_by_name = {}
    for model_name, model_config in config['policy_model'].items():
        """
        path_to_model,
        path_to_chat_template,
        api_key,
        gpu,
        port=8000
        """
        client, process = build_client(model_config['path_to_model'],
                     model_config['path_to_chat_template'],
                     model_config['api_key'],
                     model_config['GPU'],
                     model_config['port']
                     )
        policy_model_by_name[model_name] = [client, process]
    
    
    
    
    return policy_model_by_name
    




def generate_vllm_api(args, prompt, client, model_config, thread_results, thread_id):
    try:
        #print(model_config)
        completion = client.chat.completions.create(
            model=model_config['path_to_model'],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=model_config['stop_tokens'],
            #stop_token_ids = [128001, 128009],
            #frequency_penalty=0,
            #presence_penalty=0,
            #stop='',
            n=1,
        )
        #import os
        #os.system('gpustat')
        #print(completion)
        response = completion.choices[0].message.content
    
    except Exception as e:
        print(f'{e}')
        response = None
    
    #print(f"\n\ n\n>>> {prompt}\n\n\n>>> {response}\n\n----------------------------------")

    thread_results[thread_id] = response

def generation_thread(args, prompts, clients, model_configs):
    all_generations = []
    for i in tqdm(range(0, len(prompts), args.parallel_num)):
        c_data_for_gen = []
        c_clients = []
        c_model_configs = []
        for j in range(args.parallel_num):
            if i + j <= len(prompts)-1:
                c_data_for_gen.append(prompts[i+j])
                c_clients.append(clients[i+j])
                c_model_configs.append(model_configs[i+j])
                

        thread_results = {}
        threads = []
        for thread_id, (prompt, c_client, c_model_config) in enumerate(zip(c_data_for_gen, c_clients, c_model_configs)):
            t = threading.Thread(
                            target=generate_vllm_api, 
                            args=(args, prompt, c_client, c_model_config, thread_results, thread_id))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        for thread_id in range(len(c_data_for_gen)):
            all_generations.append(
                thread_results[thread_id]
            )
    return all_generations


def extract_gsm_answer(args):
    ## sample initial responses, each model sample N responses
    # read config file, which contains the info of models
    with open(args.path_to_config, 'r') as f:
        configs = f.read()
        configs = json.loads(configs)
    model_names = [ key for key in configs['policy_model'].keys()]
    print(model_names)

    policy_model_by_name = load_models(configs)

    
    # read input prompts
    all_input_items, all_responses = [], []
    with open(args.input, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            all_input_items.append(item)
            all_responses.append(item['best_response'])
    
    template = "Extract the final answer from the following content:\n\n{}"
    packed_inputs = [ template.format(r) for r in  all_responses]
    
    
    
    answers = generation_thread(args, packed_inputs, [policy_model_by_name[model_names[0]]] * len(all_input_items), [configs['policy_model'][model_names[0]]] * len(all_input_items))

    all_new_items = []
    for item, c_a in zip(all_input_items, answers):
        item['extracted_answer'] = c_a
        all_new_items.append(item)
    

        
    ## save outputs
    save_data_to_json(all_new_items, args.output)
    
    
    
    

def main(
):
    parser = argparse.ArgumentParser(prog='Generate', description='Generate responses on the eval set')
    
    ## I / O params
    parser.add_argument('--input', type=str, required=True, help="path to input data, in .jsonl")
    parser.add_argument('--output', type=str, required=True, help="path to save data, in .jsonl")

    ## model params
    #parser.add_argument('--parallel_num', type=int, required=True, default=100, help="number of threads for generation per model")
    parser.add_argument('--path_to_config', type=str, required=True, help="path to config file for models")
    parser.add_argument('--parallel_num', type=int, required=False, default=500, help="number of threads")
    

    ## sampling params
    parser.add_argument('--max_tokens', type=int, default=1024, required=False, help="")
    parser.add_argument('--temperature', type=float, default=0.0, required=False, help="")
    parser.add_argument('--top_p', type=float, default=0.7, required=False, help="")
    


    args = parser.parse_args()
    
    
    extract_gsm_answer(args)
    




   
if __name__ == "__main__":
    main()
