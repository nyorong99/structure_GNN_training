# -*- coding: utf-8 -*-
import os, random, numpy as np
try:
    import torch
except Exception:
    torch = None

def setup_seed(seed: int = 1):
    """Minimal reproducibility across python/numpy/torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
