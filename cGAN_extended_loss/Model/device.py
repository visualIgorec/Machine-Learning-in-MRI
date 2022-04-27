import torch

def device():
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
    else:
        device = "cpu"
        
    return device