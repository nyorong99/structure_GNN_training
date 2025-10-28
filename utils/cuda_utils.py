# -*- coding: utf-8 -*-
try:
    import torch
except Exception:
    torch = None

def print_cuda_memory(device: int = 0):
    """Very small CUDA memory snapshot."""
    if torch is None or not torch.cuda.is_available():
        print("\n[CUDA] not available.")
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserv = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_res = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    print(f"\nCUDA:{device}")
    print(f"torch.cuda.memory_allocated   : {alloc:.3f}GB")
    print(f"torch.cuda.memory_reserved    : {reserv:.3f}GB")
    print(f"torch.cuda.max_memory_reserved: {max_res:.3f}GB")
