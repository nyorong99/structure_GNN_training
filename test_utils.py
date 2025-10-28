# -*- coding: utf-8 -*-
"""
Quick sanity check for utils/ module imports
"""
from utils.logger import print_log, is_rank0
from utils.random_seed import setup_seed
from utils.nn_utils import count_parameters
from utils.cuda_utils import print_cuda_memory
import torch
import torch.nn as nn


def main():
    # 1️⃣ logging
    print_log("=== utils import test ===")
    print_log(f"is_rank0() = {is_rank0()}")

    # 2️⃣ seed
    setup_seed(1234)
    print_log("Seed fixed to 1234")

    # 3️⃣ parameter counting
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    n_params = count_parameters(model)
    print_log(f"Trainable parameters: {n_params:,}")

    # 4️⃣ CUDA memory (if GPU available)
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        print_cuda_memory()
        del x
        torch.cuda.empty_cache()
    else:
        print_log("CUDA not available")

    print_log("=== test complete ===")


if __name__ == "__main__":
    main()
