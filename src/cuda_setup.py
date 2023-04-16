import torch
import logging as log

def load_device():
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    # print(device)
    log.info(f"Using {device}")
    return device

def set_seeds():
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = True #as True useful with training CNN networks