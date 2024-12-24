from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from comet import load_from_checkpoint
import sys
import socket
import uuid
import json
import os
import threading


def generate_uuid():
    # 生成一个随机的 UUID 字符串
    return str(uuid.uuid4())


def write_config(config, model_name, port, host, cuda, root_to_save):
    host = socket.gethostname()
    
    new_config = {
        "model_name": model_name,
        "port": port,
        "host": host
    }
    
    
    file_name = f"{host}.model={model_name}.port={port}.cuda={cuda}.json"
    path_to_save = os.path.join(root_to_save, file_name)
    json.dump(new_config, open(path_to_save, 'w'), indent=4)
    
    print (f"config saved to ---> {path_to_save}")

app = FastAPI()

class EndpointHandler():
    def __init__(self, path="",cuda = '0'):
        self.model = load_from_checkpoint(path)
        self.model.to(f"cuda:{cuda}")
        self.lock = threading.Lock()
        self.cuda = int(cuda)

    def __call__(self, data: Dict[str, Any]) -> List[Any]:
        inputs = data.pop("inputs")
        batch_size = inputs.pop("batch_size")
        workers = inputs.pop("workers")
        data = inputs.pop("data")
        
        # model_output = self.model.predict(data, batch_size=batch_size, num_workers=workers, gpus=1)
        # scores = model_output["scores"]
        # return scores
        with self.lock:
            model_output = self.model.predict(data, batch_size=batch_size, num_workers=workers, gpus=1, devices = [self.cuda])
            scores = model_output["scores"]
            return scores


# 实例化你的模型处理器
path_to_config = sys.argv[1]
cuda = sys.argv[2]

with open(path_to_config, 'r') as f:
        config = json.load(f)
        print (f"\n\n{config}\n\n")
        
        reward_model = config['reward_model']
        model_path = config['model_path']
        
        #model_name = list(config['reward_model'].keys())[0]
        #print (f"\n\n{model_name}\n\n")
        #model_path = config['reward_model']['model_path']
handler = EndpointHandler(path=model_path, cuda=cuda)
#handler = EndpointHandler(path="../models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt")

class PredictionRequest(BaseModel):
    inputs: Dict[str, Any]

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        result = handler(request.dict())
        return {"scores": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print (f"Please provide: path_to_config, cuda id, root_to_save, port\n\n")
    
    root_to_save = sys.argv[3]
    hostname = socket.gethostname()
    port = int(sys.argv[4])
    write_config(config, reward_model, port, hostname, cuda, root_to_save)
    uvicorn.run(app, host=hostname, port=port)
    
    
    
    