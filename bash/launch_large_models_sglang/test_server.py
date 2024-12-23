import argparse
import jsonlines
from tqdm import tqdm
import json
import threading
import numpy as np
import subprocess
from openai import OpenAI
import random
import sys
import os

def list_files_and_paths(directory):
    # 创建一个空列表来保存文件名和它们的绝对路径
    files_list = []
    
    # 使用 os.listdir() 获取目录下所有文件和文件夹的名称
    for filename in os.listdir(directory):
        # 使用 os.path.join() 获取完整的文件路径
        filepath = os.path.join(directory, filename)
        
        # 检查这个路径是否真的指向一个文件
        if os.path.isfile(filepath):
            # 使用 os.path.realpath() 获取文件的绝对路径（解析任何符号链接）
            real_path = os.path.realpath(filepath)
            # 将文件名和绝对路径作为一个元组添加到列表中
            #files_list.append((filename, real_path))
            files_list.append(real_path)
    
    return files_list

## =======>>>> Start Server
def test_client(
  path_to_model,
  path_to_chat_template,
  api_key,
  gpu,
  gpu_utilize,
  port=8000,
  stop_tokens=None,
  host='localhost'
):
    ## load policy models
    #gpu = '0'  # GPU编号
    #path_to_model = '/mnt/data/haiye/llms/Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa'  # 模型路径
    #api_key='token-abc123'
    #path_to_chat_template='/mnt/data/haiye/ensemble_inference/ensemble_inference/chat_templates/llama-3-instruct.jinja'
    # 构建命令
    try:
        print (f"http://{host}:{port}/v1")
        print (f"{api_key}")
        client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key,
        )
        completion = client.chat.completions.create(
                model=path_to_model,
                messages=[{"role": "user", "content": "who are you? one word to answer" }],
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stop=stop_tokens, #"<|eot_id|>",
                n=1,
            )
        print (completion)
        response = completion.choices[0].message.content.strip()
        print (f"\n\n----> {path_to_model}\nOutput:\n{response}")
        return 1
    except Exception as e:
        print(f"{e}")
    return 0
    


def build_and_store_model(model_name, model_config):
    if "host" in model_config:
        return test_client(
            model_config['path_to_model'],
            model_config['path_to_chat_template'],
            model_config['api_key'],
            model_config['GPU'],
            model_config['gpu_utilize'],
            model_config['port'],
            model_config['stop_tokens'],
            model_config['host']
            
        )
    else:
        return test_client(
            model_config['path_to_model'],
            model_config['path_to_chat_template'],
            model_config['api_key'],
            model_config['GPU'],
            model_config['gpu_utilize'],
            model_config['stop_tokens'],
            model_config['port']
        )
        

if __name__ == "__main__":
    root_path = sys.argv[1] #'server_configs'
    # read all configs from the root path
    config_paths = list_files_and_paths(root_path)
    model_configs = []
    
    for path in config_paths:
        with open(path, 'r') as f:
            config = json.load(f)
            model_configs.append(
                (config['model_name'], config['config'])
            )
    

    result = ''
    for model_name, model_config in model_configs:
        host = model_config['host']
        gpu = model_config['GPU']
        print (model_config)
        status = build_and_store_model(model_name, model_config)
        if status:
            result += f"\n[{model_name}] in [{host}] with GPU [{gpu}] is successful!"
        else:
            result += f"\n** [{model_name}] in [{host}] with GPU [{gpu}] failed!"
        
    print (f"\n\n\n{result}")
