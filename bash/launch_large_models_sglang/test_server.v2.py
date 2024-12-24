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

    system_prompt = """
        You are a skilled editor tasked with refining a reference answer. Your goal is to enhance the answer to better align with the user's preferences by improving aspects such as accuracy and coherence. Your answer should be self-contained and not mention that your response is a refined version.
        
        The reference answer is:
        The idea that the moon covers every point on Earth in a northward progression with each moonrise is a common misunderstanding. In actuality, the moon's visibility from Earth does not follow a sequential or systematic coverage of points. Instead, the moon orbits Earth in a path inclined to the Earth's equator by about 5 degrees. This orbital plane is relatively stable, and the moon consistently moves along this path.\n\nThe moon's apparent motion in the sky results from the Earth's rotation on its axis. This rotation causes the moon (and sun) to rise in the east and set in the west from any given point on Earth's surface. As the Earth rotates, different parts of the Earth's surface will see the moon at different times, but this does not mean the moon itself is moving northward point by point across the Earth.\n\nHowever, the moon does have nodes where its orbit intersects the Earth's orbital plane (the ecliptic). These nodes regress (move westward along the ecliptic) due to gravitational perturbations, completing a full cycle approximately every 18.6 years. This regression can give the illusion that the moon's path in the sky is shifting, but it does not mean that the moon is systematically covering every point on Earth's surface in a northward direction.\n\nThe apparent position of the moon at moonrise for any given location depends on the time of year, the latitude of the observer, and the phase of the moon. The moon rises roughly 50 minutes later each day on average, and its path across the sky changes throughout the month due to its orbit around Earth.\n\nTo clarify, there is no specific \"next point to the north\" that the moon will cover at the next moonrise. The moon will rise in the east, roughly along the same north-south line that it always does for a given latitude, and its position relative to the background stars will have shifted slightly eastward due to its own orbit around Earth. The distance it appears to move to the north or south is a function of the observer's latitude and the moon's declination (its position north or south of the celestial equator), and this does not change significantly from one moonrise to the next.
    """
    user_prompt = "Over two thousand years, the moon has covered every point on the Earth once. At the most recent time of a moonrise, the moon is going to cover the next point to the north. How far north will it be?"

    try:
        client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key,
        )
        completion = client.chat.completions.create(
                model=path_to_model,
                #messages=[{"role": "user", "content": "Hi?" }],
                messages = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stop=stop_tokens, #"<|eot_id|>",
                n=1,
            )
        #print (completion)
        response = completion.choices[0].message.content.strip()
        print (f"\n\n----> {path_to_model}\nOutput:\n{response}\n\n\n")
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
    root_path = 'server_configs'
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
