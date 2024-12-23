import argparse
import jsonlines
from tqdm import tqdm
import json
import threading
import numpy as np
from reward import UltraRM, ArmoRM, Shepherd_MATH_PRM, InternRM, Qwen25_Math_RM, EmptyRM
from openai import OpenAI
import random
#from monitor import monitor
import sys
import os
import sys
import queue
import openai
import time
from ensemble_methods import *


def list_files_and_paths(directory):
    files_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            real_path = os.path.realpath(filepath)
            files_list.append(real_path)
    
    return files_list

def save_data_to_json(strings, file_name, mode='w'):
    with open(file_name, mode, encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def writer(output_queue, file_name):
    with open(file_name, 'a') as file:
        writer = jsonlines.Writer(file)
        while True:
            # 从队列中获取数据，如果队列为空，则阻塞等待
            data = output_queue.get()
            if data == "DONE":
                break
            writer.write(data)
            # 确保及时写入文件
            writer._fp.flush()
            # 标记任务已完成
            output_queue.task_done()

def load_reward_model(config):
    gpu_ids = config['reward_model']['GPU']
    if config['reward_model']['name'] == 'UltraRM':
        return UltraRM(config['reward_model']['path'], gpu_ids)
    elif config['reward_model']['name'] == 'ArmoRM':
        return ArmoRM(config['reward_model']['path'], gpu_ids)
    elif config['reward_model']['name'] == 'Shepherd_MATH_PRM':
        return Shepherd_MATH_PRM(config['reward_model']['path'], gpu_ids)
    elif config['reward_model']['name'] == 'InternRM':
        return InternRM(config['reward_model']['path'], gpu_ids)
    elif config['reward_model']['name'] == 'Qwen25_Math_RM':
        return Qwen25_Math_RM(config['reward_model']['path'], gpu_ids)
    else:
        return EmptyRM()
    return None

def load_models(args):
    # return: config, clients_by_name, reward_model
    path_to_configs = list_files_and_paths(args.root_configs)
    model_names = []
    configs = {}
    policy_model_by_name = {}
    
    for path in path_to_configs:
        with open(path, 'r') as f:
            c_config = json.load(f)
            model_name = c_config['model_name']
            
            if model_name not in model_names:
                model_names.append(model_name)
            if model_name not in configs:
                configs[model_name] = []
            
            configs[model_name].append(c_config['config'])
            
            host = c_config['config']['host']
            port = c_config['config']['port']
            api_key = c_config['config']['api_key']
            c_client = OpenAI(
                base_url=f"http://{host}:{port}/v1",
                api_key=api_key,
            )
            if model_name not in policy_model_by_name:
                policy_model_by_name[model_name] = []
            policy_model_by_name[model_name].append(c_client)

    ## load the reward model
    with open(args.path_reward_config, 'r') as f:
        reward_config = json.load(f)
    
    
    # Load reward model (assuming this does not need threading for simplicity, but could be threaded similarly)
    reward_model = load_reward_model(reward_config)
    
    print (f"\n\n-------> Loaded model configs:\n{configs}\n\n")
    
    return model_names, configs, policy_model_by_name, reward_model


def thread_manager(args, task_queue, policy_model_by_name, model_names, configs, reward_model, output_queue, n_threads):
    threads = []
    for _ in range(n_threads):
        if args.mode == 'ensemble_sample_N':
            thread_method = sample_N_thread
        elif args.mode == 'ensemble_sample_N_MCTS':
            thread_method = MCTS_thread
        elif args.mode == 'ensemble_MoA':
            thread_method = MoA_thread
        elif args.mode == 'PRS':
            thread_method = PRS_thread
        elif args.mode == 'ensemble_Seq':
            thread_method = Seq_thread
        thread = threading.Thread(target=thread_method, args=(args, task_queue, model_names, configs, policy_model_by_name, reward_model, output_queue)) 
        
        thread.start()
        threads.append(thread)
    
    return threads
    

def ensemble_sampling(args, all_input_items, model_names, configs, policy_model_by_name, reward_model):    
    # 创建任务队列并填充任务
    task_queue = queue.Queue()
    for input_item in all_input_items:
        task_queue.put(input_item)
    
    # 启动写文件的线程
    output_queue = queue.Queue()
    thread_writer = threading.Thread(target=writer, args=(output_queue, args.output))
    thread_writer.start()
    
    # 启动线程管理器
    print (f'\n\nStart the job ...')
    threads = thread_manager(args, task_queue, policy_model_by_name, model_names, configs, reward_model,output_queue, args.parallel_num)
    
    
    # 等待所有任务完成
    task_queue.join()
    # 等待所有线程自然结束
    for thread in threads:
        thread.join()
            
    output_queue.join()
    output_queue.put("DONE") 
    thread_writer.join()


def main(
    stop_event
):
    parser = argparse.ArgumentParser(prog='Generate', description='Generate responses on the eval set')
    ## mode
    parser.add_argument('--mode', type=str, required=True, choices= ['ensemble_sample_N', 'ensemble_sample_N_UCB', 'ensemble_sample_N_MCTS', 'ensemble_Seq', 'ensemble_MoA', 'PRS', ], help="sampling mode")
    
    ## I / O params
    parser.add_argument('--input', type=str, required=True, help="path to input data, in .jsonl")
    parser.add_argument('--output', type=str, required=True, help="path to save data, in .jsonl")
    parser.add_argument('--save_mode', type=str, required=False, default='w', choices=['w', 'a'], help="the mode for saving the data")
    parser.add_argument('--root_configs', type=str, required=True, help="path to the dir of model configs")
    parser.add_argument('--path_reward_config', type=str, required=True, help="path to the reward model")


    ## model params
    parser.add_argument('--path_to_config', type=str, required=False, help="path to config file for models")
    parser.add_argument('--n_samples', type=int, required=True, default=1, help="the number of samples to generate per model")
    parser.add_argument('--parallel_num', type=int, required=False, default=500, help="number of threads")
    parser.add_argument('--batch_size', type=int, required=False, default=1000, help="steps to save data")
    
    ## sampling params
    parser.add_argument('--max_tokens', type=int, default=1024, required=False, help="")
    parser.add_argument('--temperature', type=float, default=0.9, required=False, help="")
    parser.add_argument('--top_p', type=float, default=0.7, required=False, help="")
    
    ## UCB and MCTS
    parser.add_argument('--tau', type=float, required=False, default=0.5 , help="tau for sigmoid")
    parser.add_argument('--alpha', type=float, required=False, default=2 , help="weight for calculating the delta used for exploration")
    parser.add_argument('--n_new_trail', type=int, required=False, default=0, help="the gap for the new trail")
    parser.add_argument('--max_num_add_new_arms', type=int, required=False, default=2, help="the maximal number of adding new arms")
    parser.add_argument('--decay', type=float, required=False, default=1 , help="decay weight for exploration: applying when adding new arms")
    parser.add_argument('--width', type=int, required=False, default=10, help="the width of the tree for MCTS")
    parser.add_argument('--topk_child', type=int, required=False, default=1, help="number of top child nodes to keep in MCTS")
    
    ## MoA
    parser.add_argument('--n_iter', type=int, required=False, default=2, help="for MoA, the num of iterations for refinement, at least >= 2")
    parser.add_argument('--path_to_aggregator_prompt', type=str, required=False, help="path the aggregator prompt")
    parser.add_argument('--num_aggregation', type=int, required=False, default=1, help="the num of responses for aggregation")


    ## refinement
    parser.add_argument('--path_to_refine_template', type=str, required=False, help="path to the template of refinement")

    args = parser.parse_args()
    ## 0. print the vars
    print('\n\n')
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print('\n\n')
    

    ## 1. load models 
    model_names, configs, policy_model_by_name, reward_model = load_models(args)
    
    
    ## 2. read input data
    ## 2.1 remove the data that has been generated.
    ids_with_response = {}
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            output_data = f.readlines()
            for item in output_data:
                item = json.loads(item)
                ids_with_response[item['id']] = None
    print(f"Data that has been generated:\n{[k for k in ids_with_response.keys()]}")
    ## 2.2
    all_input_items = []
    with open(args.input, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            if item['id'] not in ids_with_response:
                all_input_items.append(item)
    
    ## 3. do sampling
    for i in tqdm(range(0, len(all_input_items), args.batch_size)):
        c_input_items_for_gen = []
        for j in range(args.batch_size):
            if i + j <= len(all_input_items)-1:
                c_input_items_for_gen.append(all_input_items[i+j])

        start_time = time.time()

        ensemble_sampling(args, c_input_items_for_gen, model_names, configs, policy_model_by_name, reward_model)

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_hours = elapsed_time / 3600
        print(f"Run Time: {elapsed_hours:.4f} hours")

    
    stop_event.set()

if __name__ == "__main__":    
    # 创建一个事件，用来通知监视线程何时停止
    stop_event = threading.Event()
    # 创建监视线程
    main(stop_event)

    