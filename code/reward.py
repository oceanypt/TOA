import argparse
import jsonlines
from tqdm import tqdm
import torch
import json
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn as nn
import torch
from typing import Optional, List
import threading
from multiprocessing import Pool
import numpy as np
import gc
from torch.nn import DataParallel
import sys
from comet import download_model, load_from_checkpoint
import requests
import random
import re
import subprocess

from math_parser import *


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        

    def get_template(self):
        return self.template
    
    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards
    

class UltraRM:
    def __init__(self, model_path, device_id=0):
        ## load the model
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaRewardModel.from_pretrained(model_path)
        self.model.half()
        self.model.to(device)
        self.model.eval()

        self.template = "Human: {instruction}\nAssistant: {response}"

    def cal_rewards(self, instructions, responses):
        prompts = [self.template.format(instruction=instruction, response=response) for instruction, response in zip(instructions, responses)]
        print(prompts[0])
        print('\n\n\n\n-------------------------')

        all_rewards = []
        for prompt in prompts:
            try:
                inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
                reward = self.model(**inputs).item()
                all_rewards.append(reward)
            except Exception as e:
                print(f"Error: {e}")
                all_rewards.append(0.0)

        return all_rewards
    
    def clear_cache(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class ArmoRM:
    def __init__(self, model_path, gpu_id):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        
        device = f'cuda:{gpu_id}'
        self.model.to(device)
        self.device = device
        
    def cal_rewards(self, instructions, responses):
        all_rewards = []
        for instruction, response in zip(instructions, responses):
            try:
                messages = [{"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}]
                input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model(input_ids)
                    preference_score = output.score.cpu().float().item()
                    all_rewards.append(preference_score)
            except Exception as e:
                print(f"\n\n\n\n---> In reward model: {e}\n\n\n\n")
                all_rewards.append(0.0)
        
        return all_rewards
    def clear_cache(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()



class XCOMET:
    def __init__(self, model_path, gpu_id):
        #self.gpu_id = gpu_id
        self.model = load_from_checkpoint(model_path)
        device = f'cuda:{gpu_id}'
        self.model.to(device)
        self.gpu_id = [ int(g) for g in gpu_id.split(',') ]
        
    def cal_rewards(self, instructions, responses, max_len=512):
        data = [{"src": src, "mt": mt[:max_len]} for src, mt in zip(instructions, responses)  ]
        
        print (f'\n\n calculating rewards in XCOMET ...\n\n')
        model_output = self.model.predict(data, batch_size=64, gpus=1, devices= self.gpu_id  )
        
        return model_output.scores
    
    def clear_cache(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


class Shepherd_MATH_PRM:
    def __init__(self, model_path, gpu_id):
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'
        
        # # Assuming 'model_path', 'good_token', and 'bad_token' are defined elsewhere in your code
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:]  # e.g., [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1]  # e.g., 12902
        self.model = AutoModelForCausalLM.from_pretrained(model_path).eval()

        
        device = f'cuda:{gpu_id}'
        self.model.to(device)
        self.device = device
        
    def cal_rewards(self, instructions, responses):
        all_rewards = []
        for instruction, response in zip(instructions, responses):
            try:
                response += ' ' + self.step_tag
                input_for_prm = f"{instruction} {response}"
                input_id = torch.tensor([self.tokenizer.encode(input_for_prm)])

                # Move the tensor to GPU
                input_id = input_id.to(self.device)

                with torch.no_grad():
                    logits = self.model(input_id).logits[:,:,self.candidate_tokens]
                    scores = logits.softmax(dim=-1)[:,:,0]
                    step_scores = scores[input_id == self.step_tag_id].tolist()
                    #print(step_scores)
                    all_rewards.append(step_scores[-1])
                    
            except Exception as e:
                print(f"\n\n\n\n---> In reward model: {e}\n\n\n\n")
                all_rewards.append(0.0)
        
        return all_rewards
    def clear_cache(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class InternRM:
    def __init__(self, model_path, gpu_id):
        # gpu_id: str, such as "0,1" or "0,1,2,3"
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map=f"cuda:{gpu_id}" #f"cuda:{gpu_id}" #'auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    def cal_rewards(self, instructions, responses):
        all_rewards = []
        for instruction, response in zip(instructions, responses):
            try:
                messages = [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}
                        ]
                score = self.model.get_score(self.tokenizer, messages)
                all_rewards.append(score)
            except Exception as e:
                print(f"\n\n\n\n---> In reward model: {e}\n\n\n\n")
                all_rewards.append(0.0)
        torch.cuda.empty_cache()
        gc.collect()
        return all_rewards
    def clear_cache(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()



class XCOMT_API:
    def __init__(self, api_urls, api_token=''):
        self.api_urls = api_urls
        #self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
    
    def cal_rewards(self, srcs, nmts):
        api_url_ids = list(range(len(self.api_urls)))
        
        data = {
                    "inputs": {
                    "batch_size": 25,
                    "workers": 1,
                    "data": [
                        {"src": src,
                        "mt": nmt} for src, nmt in zip(srcs, nmts)
                        ]
                    }
                }
        
        random.shuffle(api_url_ids)
        print (f'\n\n--> xcomet id: {self.api_urls[api_url_ids[-1]]}')
        try:
            response = requests.post(self.api_urls[api_url_ids[-1]], headers=self.headers, json=data)
            result = response.json()  # 获取并解析响应
            print (result)
            return result['scores']
        except Exception as e:
            print(f"{e}")
            return [0.0] * len(srcs)
            
    
        
     


class Qwen25_Math_RM:
    def __init__(self, model_path, gpu_id):
        # gpu_id: str, such as "0,1" or "0,1,2,3"
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
            device_map='auto' #f"cuda:{gpu_id}" #f"cuda:{gpu_id}" #'auto'
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    def cal_rewards(self, instructions, responses):
        all_rewards = []
        for instruction, response in zip(instructions, responses):
            try:
                messages = [
                        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}
                        ]
                conversation_str = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                print (f"******* {self.model.device}")
                input_ids = self.tokenizer.encode(
                    conversation_str, 
                    return_tensors="pt", 
                    add_special_tokens=False
                ).to(self.model.device)
                
                score = self.model(input_ids=input_ids)[0].cpu().float().item()
                print (f"********** reward: {score}")
                all_rewards.append(score)
            
            except Exception as e:
                print(f"\n\n\n\n---> In reward model: {e}\n\n\n\n")
                all_rewards.append(0.0)

        return all_rewards

        
        
        
        
class SkyworkRM:
    def __init__(self, model_path, gpu_id):
        # gpu_id: str, such as "0,1" or "0,1,2,3"
        self.device = f"cuda:{gpu_id}"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        
    def cal_rewards(self, instructions, responses):
        all_rewards = []
        for instruction, response in zip(instructions, responses):
            try:
                messages = [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response}
                        ]
                messages_tokenized = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    score = self.model(messages_tokenized).logits[0][0].item()
                all_rewards.append(score)
            except Exception as e:
                print(f"\n\n\n\n---> In reward model: {e}\n\n\n\n")
                all_rewards.append(0.0)

        return all_rewards

    
class EmptyRM:
    def __init__(self, model_path=None, gpu_id=None):
        pass
    
    def cal_rewards(self, instructions, responses):
        all_rewards = []
        for instruction, response in zip(instructions, responses):
            all_rewards.append(0.0)

        return all_rewards

