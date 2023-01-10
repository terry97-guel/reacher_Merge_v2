### Casting ###
import numpy as np
import torch
import random

def cast_tensor(array):
    if isinstance(array, torch.Tensor): return array
    else: torch.tensor(array)

def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np

def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

### Logging ###
from datetime import datetime

def get_runname():
    now = datetime.now()
    format = "%m%d:%H%M"
    runname = now.strftime(format)
    return runname
    

def print_log_dict(log_dictionary):
    log_strs = ""
    for key, item in log_dictionary.items():
        log_str = "{}:{:.2f}, ".format(key,item)
        log_strs = log_strs + log_str
    print(log_strs, "\n")    


def prefix_dict(log_dictionary:dict, prefix:str):
    return {prefix+"_"+key: value for key,value in log_dictionary.items()}

### Consturcting ###
from torch import nn

def get_linear_layer(hdim, hidden_actv, std=0.1):
    layers = []
    for hdim_idx in range(0,len(hdim)-1):
        layer = nn.Linear(hdim[hdim_idx],hdim[hdim_idx+1])
        # torch.nn.init.normal_(layer.weight,.0,std)
        torch.nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
        layers.append(hidden_actv())
    return layers

### Seed ###
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determsinistic = True
    torch.backends.cudnn.benchmark = False
    

def set_wandb(pname=None, runname=None):
    import wandb
    if pname is not None:
        wandb.init(project = pname)
    else:
        wandb.init()
        
    if runname is not None:
        wandb.run.name = runname
        
        
### Visualize ###
def make_circle(radius, center=np.zeros(2)):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius * np.cos(theta) + center[0]
    b = radius * np.sin(theta) + center[1]
    return [a,b]





