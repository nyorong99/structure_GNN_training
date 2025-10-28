# -*- coding: utf-8 -*-
import os, time
try:
    import torch.distributed as dist
except Exception:
    dist = None

_LOG_FILE = None

def is_rank0() -> bool:
    if dist and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    r = os.environ.get("RANK")
    return (r is None) or (r == "0")

def set_log_file(path: str):
    """Optional: call once to also log to a file."""
    global _LOG_FILE
    _LOG_FILE = path
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def print_log(msg):
    """Tiny logger: prints timestamped line (rank0 only)."""
    if not is_rank0():
        return
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line)
    if _LOG_FILE:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
