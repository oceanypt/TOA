
# python start_server.py $path_to_config


python start_server.py ../../model_configs/llama-3.1-70b-instruct.json ./server_configs 4,5,6,7 8001 0.9


python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 0,1,2,3 8000 0.9

python start_server.py ../../model_configs/Mixtral-8x22B-Instruct-v0.1.json ./server_configs 0,1,2,3 8000 0.9


python start_server.py  ../../model_configs/Yi-1.5-34b-chat-16k.json  ./server_configs 0 8000 0.9
 

python start_server.py  ../../model_configs/Yi-1.5-34b-chat-16k.json  ./server_configs 4 8001 0.9 & python start_server.py  ../../model_configs/Yi-1.5-34b-chat-16k.json  ./server_configs 5 8002 0.9 & python start_server.py  ../../model_configs/Yi-1.5-34b-chat-16k.json  ./server_configs 6 8003 0.9 & python start_server.py  ../../model_configs/Yi-1.5-34b-chat-16k.json  ./server_configs 7 8004 0.9



python start_server.py ../../model_configs/Mixtral-8x22B-Instruct-v0.1.json ./server_configs 0,1,2,3 8000 0.9 & python start_server.py ../../model_configs/Mixtral-8x22B-Instruct-v0.1.json ./server_configs 4,5,6,7 8001 0.9

python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 0,1,2,3 8000 0.9 & python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 4,5,6,7 8001 0.9



python start_server.py ../../model_configs/wizardlm-2-8x22b.json ./server_configs 0,1,2,3 8000 0.9


python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 0,1,2,3 8000 0.9  &  python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 4,5,6,7 8001 0.9  &


python start_server.py ../../model_configs/mistral-large-instruct-2407.json ./server_configs 0,1,2,3,4,5,6,7 8000 0.9 


python start_server.py ../../model_configs/llama-3.1-70b-instruct.json ./server_configs 0,1,2,3 8000 0.9



python start_server.py ../../model_configs/wizardlm-2-8x22b.json ./server_configs 0,1,2,3 8000 0.9 
python start_server.py ../../model_configs/wizardlm-2-8x22b.json ./server_configs 4,5,6,7 8001 0.9 

python start_server.py ../../model_configs/llama-3.1-70b-instruct.json ./server_configs 0,1,2,3 8000 0.9
python start_server.py ../../model_configs/llama-3.1-70b-instruct.json ./server_configs 4,5,6,7 8001 0.9


python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 0,1,2,3 8000 0.9 
python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 4,5,6,7 8001 0.9


python start_server.py ../../model_configs/Mixtral-8x22B-Instruct-v0.1.json ./server_configs 0,1,2,3 8000 0.9
python start_server.py ../../model_configs/Mixtral-8x22B-Instruct-v0.1.json ./server_configs 4,5,6,7 8001 0.9


python start_server.py ../../model_configs/mistral-large-instruct-2407.json ./server_configs 0,1,2,3 8000 0.9 
python start_server.py ../../model_configs/mistral-large-instruct-2407.json ./server_configs 4,5,6,7 8001 0.9



python start_server.py ../../model_configs/wizardlm-2-8x22b.json ./server_configs 0,1,2,3,4,5,6,7 8000 0.85

python start_server.py ../../model_configs/llama-3.1-70b-instruct.json ./server_configs 0,1,2,3,4,5,6,7 8000 0.85


python start_server.py ../../model_configs/Qwen2-72B-Instruct.json ./server_configs 0,1,2,3,4,5,6,7 8000 0.85


python start_server.py ../../model_configs/Mixtral-8x22B-Instruct-v0.1.json ./server_configs 0,1,2,3,4,5,6,7 8000 0.85


python start_server.py ../../model_configs/mistral-large-instruct-2407.json ./server_configs 0,1,2,3,4,5,6,7 8000 0.85


. ~/anaconda3/bin/activate
conda activate /root/anaconda3/envs/sglang
pip install jsonlines
pip install "unbabel-comet>=2.2.0"
pip install modelscope
export VLLM_USE_MODELSCOPE=False
export MKL_SERVICE_FORCE_INTEL=1
export HF_ENDPOINT=https://hf-mirror.com






