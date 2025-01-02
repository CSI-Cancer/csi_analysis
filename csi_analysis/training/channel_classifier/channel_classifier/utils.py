import numpy as np
import torch
import random


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Parameters:
    - seed: The seed value to set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False