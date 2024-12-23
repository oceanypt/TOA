from openai import OpenAI
import subprocess
import sys
import json
import socket
import os
import uuid
import time

def generate_uuid():
    # 生成一个随机的 UUID 字符串
    return str(uuid.uuid4())


def get_first_ip_address():
    # 执行 hostname -I 命令
    result = subprocess.run(['hostname', '-I'], stdout=subprocess.PIPE, text=True)
    
    # 获取命令输出并分割为列表
    ip_addresses = result.stdout.strip().split()
    
    # 返回列表中的第一个 IP 地址
    return ip_addresses[0] if ip_addresses else None

## read the model config
## 1. launch the model 2. return the model config: hostname, etc


def build_client(
  path_to_model,
  path_to_chat_template,
  api_key,
  gpu,
  port=8000,
  gpu_utilize=0.9,
  stop_tokens=None
):  
    ## get the hostname
    #host = get_first_ip_address() #socket.gethostname()
    host = socket.gethostname()
    
    try:
        client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
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
        print (f'Start the server successfully.')
    except Exception as e:
        print(f"{e}")
    
    
    gpu_num = len(gpu.split(','))
    
    
    #command = f"CUDA_VISIBLE_DEVICES={gpu} nohup python -m vllm.entrypoints.openai.api_server " \
    #      f"--host {host} " \
    #      f"--model {path_to_model} --dtype auto " \
    #      f"--api-key {api_key} " \
    #      f"--port {port} --chat-template {path_to_chat_template} --disable-log-stats --tensor-parallel-size {gpu_num} --max_model_len 15000 --gpu_memory_utilization {gpu_utilize} --trust-remote-code > log.launch &"
    
    # command = f"""
    #   CUDA_VISIBLE_DEVICES={gpu}  python -m sglang.launch_server --model-path {path_to_model} --api-key {api_key} --host {host} --port {port} --tp {gpu_num} --chat-template {path_to_chat_template} --trust-remote-code  --dtype auto  --context-length 15000 --enable-p2p-check  --mem-fraction-static {gpu_utilize}
    # """
    
    command = f"""
         CUDA_VISIBLE_DEVICES={gpu} nohup  python -m sglang.launch_server --model-path {path_to_model} --api-key {api_key} --host {host} --port {port} --tp {gpu_num} --chat-template {path_to_chat_template} --trust-remote-code  --dtype auto  --context-length 15000 --enable-p2p-check  --mem-fraction-static {gpu_utilize} > log.launch &
     """
    
    

    print (f'\n\n{command}\n\n')
    
    process = subprocess.Popen(command, shell=True) # keep process to kill in the end

    client = OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key=api_key,
    )
    while True:
        try:
            client.chat.completions.create(
                model=path_to_model,
                messages=[{"role": "user", "content": "Hi!" }],
                max_tokens=15,
                temperature=0.7,
                top_p=0.9,
                stop=stop_tokens, #"<|eot_id|>",
                n=1,
            )
            break
        except Exception as e:
            pass
    print (f'Start the server successfully.')
    

def write_config(config, model_name, api_key, port, GPU, gpu_utilize, root_to_save):
    #host = get_first_ip_address() #socket.gethostname()
    host = socket.gethostname()
    
    new_config = {
        "model_name": model_name,
        "config": {
            "path_to_model": config['path_to_model'],
            "path_to_chat_template": config['path_to_chat_template'],
            "stop_tokens": config['stop_tokens'],
            "api_key": api_key,
            "port": port,
            "host": host,
            "GPU": GPU,
            "gpu_utilize": gpu_utilize
        }
    }
    
    gpus = '_'.join(GPU.split(','))
    
    file_name = f"{host}.model={model_name}.gpu={gpus}.port={port}.json"
    path_to_save = os.path.join(root_to_save, file_name)
    json.dump(new_config, open(path_to_save, 'w'), indent=4)
    
    print (f"config saved to ---> {path_to_save}")
    


if __name__ == "__main__":
    print (f"Please specify:\n1. path_to_config\n2. root_to_save\n3. GPU\n4. port: make sure models use different port number in the same node.\n5. gpu_utilize\n6. The key is randomly generated")
    
    path_to_config = sys.argv[1]
    root_to_save = sys.argv[2]
    GPU = sys.argv[3]
    port = int(sys.argv[4])
    gpu_utilize= float(sys.argv[5])
    
    
    with open(path_to_config, 'r') as f:
        config = json.load(f)
        print (f"\n\n{config}\n\n")
        model_name = list(config['policy_model'].keys())[0]
        print (f"\n\n{model_name}\n\n")
    
    #api_key = generate_uuid()
    api_key = 'abc123'
    
    kill_command = """
                pkill -9 -f "python -m sglang.launch_server" &
                pkill -9 -f "/root/anaconda3/envs/sglang/bin/python" & 
                pkill -9 -f "python -m vllm.entrypoints.openai.api_server" &
                pkill -9 -f "/root/anaconda3/envs/llama31_infer/bin/python" &
                pkill -9 -f "/root/anaconda3/envs/llama31_infer/bin/python -c from multiprocessing" &
        """
    # subprocess.Popen(kill_command, shell=True)
    # ## 0. start the model
    # build_client(path_to_model = config['policy_model'][model_name]['path_to_model'],
    #              path_to_chat_template = config['policy_model'][model_name]['path_to_chat_template'],
    #              api_key = api_key,
    #              gpu = GPU,
    #              port = port,
    #              gpu_utilize = gpu_utilize,
    #              stop_tokens = config['policy_model'][model_name]['stop_tokens'],
    #             )
    # write_config(config['policy_model'][model_name], model_name, api_key, port, GPU, gpu_utilize, root_to_save)
    
    interval = 12 * 60 + 30
    while True:
        ## 1. kill the server
        subprocess.Popen(kill_command, shell=True)
        
        ## 2. start the model
        build_client(path_to_model = config['policy_model'][model_name]['path_to_model'],
                 path_to_chat_template = config['policy_model'][model_name]['path_to_chat_template'],
                 api_key = api_key,
                 gpu = GPU,
                 port = port,
                 gpu_utilize = gpu_utilize,
                 stop_tokens = config['policy_model'][model_name]['stop_tokens'],
                )

        write_config(config['policy_model'][model_name], model_name, api_key, port, GPU, gpu_utilize, root_to_save)
        time.sleep(interval)

    
    
    
    
    
  
    
    
