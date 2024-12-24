# TOA: 


<!-- <p align="center">
  <!-- <em></em>
  <br>
   -->
  <!-- <img src="./figures/mas.png" alt="" width="400">
</p> -->

<!-- [![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/pdf/2412.17061)
   -->

<div style="text-align: center;">
  <img src="./figures/mas.png" alt="" width="400">
  <br>
  <a href="https://arxiv.org/pdf/2412.17061" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/arXiv-paper-b31b1b.svg" alt="arXiv Paper">
  </a>
</div>

<br>

This is officical repository for the work [Multi-Agent Sampling: Scaling Inference Compute for Data Synthesis with Tree Search-Based Agentic Collaboration](https://arxiv.org/pdf/2412.17061). We study how to synthesize data for alignment from multiple distinct language models such as Llama3, Qwen2, Mistral, etc, which is so called problem of multi-agent sampling. We propose [TOA]() (Tree Search-based Orchestrated Agents) to achieve this goal. Our method is driven by Monte Carlo Tree Search with a Reward Model integrated. 


**TOA** is designed to **synthesize alignment data** (specifically the output responses) from a diverse range of language models. 


## 🌟 Key Features

- 🔓 **Open-source models**: [Llama Series](https://huggingface.co/meta-llama), [Qwen Series](https://huggingface.co/Qwen), [Mistral Series](https://huggingface.co/mistralai), and more.
- 🔒 **Closed-source models**: OpenAI, Claude, etc.
- 😊 **OpenAI Compatible Server**: We support OpenAI compatible API to use the models. 
- 🎯 **Reward Model Integration**: TOA utilizes a reward model to guide and optimize the generation process. You can easily specifiy a your own reward model.
- 💰 **Compute Efficient**: For each input question, TOA optimizes the generation structure dynamically with MCTS-based search, making our method more compute-efficient than other baselines for data synthesis. 









## News
- [2024/12/22] [TOA paper](https://arxiv.org/pdf/2412.17061) is out at arXiv. 



## Quick Start

### 1. Start Local Servers
If you want to host the language models locally, you can use the provide the code to start the local servers. 

```bash
cd bash/launch_large_models

python start_server.vllm.py path_to_config root_to_save GPU port gpu_utilize
```


- path_to_config: path to the configuration file of the model, which is in JSON format and looks like
```bash
{
    "policy_model": {
            "llama-3.1-8b-instruct": {
                "path_to_model": "",
                "path_to_chat_template": "../chat_templates/llama-3.1-instruct.jinja",
                "stop_tokens": "['<|eot_id|>']"
        }
    }
}
```
- root_to_save: path to save the server configuration, which is in JSON format and looks like:
```bash

    "model_name": "llama-3.1-8b-instruct",
    "config": {
        "path_to_model": "",
        "path_to_chat_template": "../chat_templates/llama-3.1-instruct.jinja",
        "stop_tokens": "['<|eot_id|>']",
        "api_key": "abc123",
        "port": 8000,
        "host": your local machine address,
        "GPU": "0",
        "gpu_utilize": 0.9
    }
}
```
- GPU: gpu ids, such as "0", "0,1", "0,1,2,3"
- port: 8000, 8001, etc
- gpu_utilize: how much gpu memory to use, such as 0.9, 0.8

You can start the server for different models, just make sure to save all the server configuration into one same folder, that is [root_to_save](). 
















![](./figures/method.png)



