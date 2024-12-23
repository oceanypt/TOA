import requests
import json
import threading
from queue import Queue
import sys
import os

def list_files_and_paths(directory):
    files_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            real_path = os.path.realpath(filepath)
            files_list.append(real_path)
    
    return files_list

def send_request(API_URL, data):
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        #print(response)
        result = response.json()  # 获取并解析响应
        print (f"\n\n{result}")
        return 1
    except Exception as e:
        print(f"{e}")
        return 0


root_reward_configs = sys.argv[1]

path_to_reward_configs = list_files_and_paths(root_reward_configs)
for path in path_to_reward_configs:
    with open(path, 'r') as f:
        reward_config = json.load(f)
        api_url = f"http://{reward_config['host']}:{reward_config['port']}/predict/"
        print(api_url)
        API_TOKEN = "YOUR_API_KEY"
        headers = {
            "Authorization": f"Bearer {'API_TOKEN'}",
            "Content-Type": "application/json",
        }
        
        data = {
            "inputs": {
                "batch_size": 1,
                "workers": 2,
                "data": [
                    {"src": "The output signal provides constant sync so the display never glitches.",
                    "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."}
                ]
            }
        }
        
        code = send_request(api_url, data)
        if code == 1:
            print (f"[success] {path}")
        else:
            print (f"**[fail] {path}")

