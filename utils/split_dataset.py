# utils/split_dataset.py
# -*- coding: utf-8 -*-
"""
Split dataset_index.jsonl into train/valid/test jsonl files if they don't exist yet.
Intended to be called at the start of training.
"""

import os
import json
import random
from typing import Tuple


def _read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("status", "success") != "success":
                continue
            items.append(obj)
    return items


def _write_jsonl(path, items):
    with open(path, "w") as f:
        for obj in items:
            f.write(json.dumps(obj) + "\n")


def prepare_splits(
    source_jsonl: str,
    out_dir: str,
    seed: int = 1234,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[str, str, str]:
    """
    Returns:
        (train_json_path, valid_json_path, test_json_path)

    Behavior:
    - If out_dir/train.jsonl etc already exist AND are non-empty, we keep them.
    - Otherwise, we generate fresh splits from source_jsonl.
    """
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.jsonl")
    valid_path = os.path.join(out_dir, "valid.jsonl")
    test_path  = os.path.join(out_dir, "test.jsonl")

    # if already exist and not empty, just return
    if (
        os.path.isfile(train_path) and os.path.getsize(train_path) > 0 and
        os.path.isfile(valid_path) and os.path.getsize(valid_path) > 0 and
        os.path.isfile(test_path)  and os.path.getsize(test_path)  > 0
    ):
        return train_path, valid_path, test_path

    # else: build fresh
    data = _read_jsonl(source_jsonl)

    # reproducible shuffle
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # assign remainder to test to avoid rounding issues
    n_test  = n - n_train - n_val
    if n_test < 0:
        # in pathological tiny dataset cases, clamp
        n_test = 0

    train_split = data[:n_train]
    val_split   = data[n_train:n_train+n_val]
    test_split  = data[n_train+n_val:n_train+n_val+n_test]

    _write_jsonl(train_path, train_split)
    _write_jsonl(valid_path, val_split)
    _write_jsonl(test_path,  test_split)

    return train_path, valid_path, test_path
