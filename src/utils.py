import torch
import numpy as np
import random

def get_device():
    """
    Returns the device to be used for PyTorch operations.
    If CUDA is available, it returns 'cuda', otherwise it returns 'cpu'.
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using device: {device} (CUDA available)")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"Using device: {device} (Apple Silicon MPS available)")
    else:
        device = 'cpu'
        print(f"Using device: {device} (CPU only)")
    return device

def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False