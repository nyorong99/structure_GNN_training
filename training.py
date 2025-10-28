# Create training.py based on the user's project structure and uploaded utils/dataset/model specs.
from pathlib import Path
# -*- coding: utf-8 -*-
"""
training.py
- Entry point for training BindingAffinityModel with PoseDataset.
- DDP/AMP compatible (torchrun optional).
- Uses small utils: utils.logger, utils.random_seed, utils.cuda_utils, utils.nn_utils

Example (single GPU):
    python training.py \
        --train_json index/train.jsonl --val_json index/valid.jsonl \
        --save_dir runs/egnn_ic50 --epochs 50 --batch_size 16

Example (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 training.py \
        --train_json index/train.jsonl --val_json index/valid.jsonl \
        --save_dir runs/egnn_ic50_ddp --epochs 50 --batch_size 32
"""

import os
import math
import json
import argparse
from dataclasses import asdict
from typing import Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from utils.logger import print_log, set_log_file, is_rank0
from utils.random_seed import setup_seed
from utils.cuda_utils import print_cuda_memory
from utils.nn_utils import count_parameters
from utils.split_dataset import prepare_splits


# Project modules
from model.model import BindingAffinityModel, ModelConfig
from model.dataset import PoseDataset, pose_collate_fn  # dataset and collate


def init_distributed() -> Dict[str, Any]:
    """Initialize torch.distributed if available (torchrun)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    use_ddp = world_size > 1

    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)

    return {
        "world_size": world_size,
        "local_rank": local_rank,
        "rank": rank,
        "use_ddp": use_ddp,
        "device": f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
    }


def cleanup_distributed(use_ddp: bool):
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Train BindingAffinityModel (EGNN-Cross)")
    # Data
    p.add_argument("--train_json", type=str, default=None, help="Path to train jsonl index. If not set, will auto-generate split from --source_index.")
    p.add_argument("--val_json", type=str, default=None, help="Path to val jsonl index. If not set, will use auto split.")
    p.add_argument("--source_index", type=str,
                   default="processed/P00533_block/dataset_index.jsonl",
                   help="Master index jsonl used to generate train/val/test splits.")
    p.add_argument("--split_dir", type=str, default="index",
                   help="Where to write/read split jsonls (train.jsonl, valid.jsonl, test.jsonl)")
    p.add_argument("--filter_all_passed", type=int, default=None, choices=[0,1],
                   help="If set, keep samples with all_passed==value")
    # I/O & logging
    p.add_argument("--save_dir", type=str, default="runs/egnn_ic50")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    # Model config
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--edge_dim", type=int, default=16)
    p.add_argument("--num_layers_prot", type=int, default=4)
    p.add_argument("--num_layers_lig", type=int, default=4)
    p.add_argument("--num_layers_cross", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (increased from 0.1 for regularization)")
    p.add_argument("--prot_in_dim", type=int, default=23)
    p.add_argument("--lig_in_dim", type=int, default=771)  # your npz currently uses 771
    p.add_argument("--use_priors", action="store_true", default=True)
    # Train setup
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization (increased from 0.0)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True, help="Use AMP (bf16 if available else fp16)")
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--save_topk", type=int, default=3, help="Keep top-k checkpoints by Val RMSE (lower is better)")
    p.add_argument("--early_stop_patience", type=int, default=30, help="Early stopping patience (epochs without improvement)")
    # Scheduler (simple cosine)
    p.add_argument("--use_cosine", action="store_true", default=True)
    p.add_argument("--cos_min_lr", type=float, default=1e-5)

    args = p.parse_args()
    return args


def make_dataloaders(args, device_info):
    train_set = PoseDataset(args.train_json, filter_all_passed=args.filter_all_passed)
    val_set = PoseDataset(args.val_json, filter_all_passed=args.filter_all_passed) if args.val_json else None

    if device_info["use_ddp"]:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False) if val_set is not None else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=pose_collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pose_collate_fn,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
        )

    return train_loader, val_loader


def build_model(args, device):
    cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        num_layers_prot=args.num_layers_prot,
        num_layers_lig=args.num_layers_lig,
        num_layers_cross=args.num_layers_cross,
        dropout=args.dropout,
        prot_in_dim=args.prot_in_dim,
        lig_in_dim=args.lig_in_dim,
        use_priors=args.use_priors,
    )
    model = BindingAffinityModel(cfg).to(device)
    return model, cfg


def compute_metrics(pred, y):
    """Return dict with rmse, mae, pearson."""
    # pred: (B,), y: (B,) or (B,1)
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)

    with torch.no_grad():
        diff = pred - y
        mse = torch.mean(diff * diff)
        rmse = torch.sqrt(mse + 1e-12)
        mae = torch.mean(torch.abs(diff))

        # Pearson
        x = pred - pred.mean()
        yy = y - y.mean()
        num = torch.sum(x * yy)
        den = torch.sqrt(torch.sum(x * x) * torch.sum(yy * yy) + 1e-12)
        pearson = num / den if den > 0 else torch.tensor(0.0, device=pred.device)

    return {
        "rmse": rmse.item(),
        "mae": mae.item(),
        "pearson": pearson.item(),
    }


def save_checkpoint(state: dict, save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ckpt_{tag}.pt")
    torch.save(state, path)
    return path


def load_best_topk(save_dir: str):
    # helper to list previous checkpoints
    if not os.path.isdir(save_dir):
        return []
    return sorted([f for f in os.listdir(save_dir) if f.startswith("ckpt_epoch") and f.endswith(".pt")])


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled, grad_clip):
    model.train()
    total_loss = 0.0
    total_n = 0
    all_pred = []
    all_y = []

    for batch in loader:
        y = batch["y"].to(device).squeeze(-1)  # (B,)
        # Model forward expects keys: prot_pos/x/edge_index/edge_attr/... and cross_attr is cross_dist
        kwargs = dict(
            prot_pos=batch["prot_pos"].to(device),
            prot_x=batch["prot_x"].to(device),
            prot_edge_index=batch["prot_edge_index"].to(device),
            prot_edge_attr=batch["prot_edge_attr"].to(device),
            prot_batch=batch["prot_batch"].to(device),
            lig_pos=batch["lig_pos"].to(device),
            lig_x=batch["lig_x"].to(device),
            lig_edge_index=batch["lig_edge_index"].to(device),
            lig_edge_attr=batch["lig_edge_attr"].to(device),
            lig_batch=batch["lig_batch"].to(device),
            cross_index=batch["cross_index"].to(device),
            cross_attr=batch["cross_dist"].to(device),
            priors=batch["priors"].to(device) if "priors" in batch else None,
            assay=None,
        )

        optimizer.zero_grad(set_to_none=True)

        # If AMP, cast all float inputs to the autocast dtype to keep h/m_ij consistent
        amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else (torch.float16 if amp_enabled else None)
        if amp_dtype is not None:
            for k, v in list(kwargs.items()):
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    kwargs[k] = v.to(amp_dtype)
            # Also convert y to the same dtype for loss calculation
            y = y.to(amp_dtype)


        if amp_enabled:
            # Prefer bf16 when available; else fall back to fp16 autocast
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                pred = model(**kwargs)  # (B,)
                loss = torch.nn.functional.mse_loss(pred, y)
            # backward -> unscale -> (optional) clip -> step -> update
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(**kwargs)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        
        # Collect predictions for metrics (detach and move to CPU)
        all_pred.append(pred.detach().cpu().float())
        all_y.append(y.detach().cpu().float())

    avg_loss = total_loss / max(total_n, 1)
    
    # Compute train metrics
    if len(all_pred) > 0:
        all_pred = torch.cat(all_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)
        train_metrics = compute_metrics(all_pred, all_y)
        train_metrics["loss"] = avg_loss
    else:
        train_metrics = {"loss": avg_loss, "rmse": float("nan"), "mae": float("nan"), "pearson": float("nan")}
    
    return train_metrics


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_pred = []
    all_y = []

    for batch in loader:
        y = batch["y"].to(device).squeeze(-1)

        kwargs = dict(
            prot_pos=batch["prot_pos"].to(device),
            prot_x=batch["prot_x"].to(device),
            prot_edge_index=batch["prot_edge_index"].to(device),
            prot_edge_attr=batch["prot_edge_attr"].to(device),
            prot_batch=batch["prot_batch"].to(device),
            lig_pos=batch["lig_pos"].to(device),
            lig_x=batch["lig_x"].to(device),
            lig_edge_index=batch["lig_edge_index"].to(device),
            lig_edge_attr=batch["lig_edge_attr"].to(device),
            lig_batch=batch["lig_batch"].to(device),
            cross_index=batch["cross_index"].to(device),
            cross_attr=batch["cross_dist"].to(device),
            priors=batch["priors"].to(device) if "priors" in batch else None,
            assay=None,
        )

        pred = model(**kwargs)
        loss = torch.nn.functional.mse_loss(pred, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())

    if total_n == 0:
        return {"loss": float("nan"), "rmse": float("nan"), "mae": float("nan"), "pearson": float("nan")}

    all_pred = torch.cat(all_pred, dim=0)
    all_y = torch.cat(all_y, dim=0)

    metrics = compute_metrics(all_pred, all_y)
    metrics["loss"] = total_loss / total_n
    return metrics


def main():
    args = parse_args()

    # DDP init
    ddp = init_distributed()
    device = torch.device(ddp["device"])

    # Save dir & logger
    if is_rank0():
        os.makedirs(args.save_dir, exist_ok=True)
        # If user didn't specify log_file, create default
        log_path = args.log_file or os.path.join(args.save_dir, "train.log")
        set_log_file(log_path)
        pretty_args = json.dumps(vars(args), indent=2, sort_keys=True)
        print_log("Args:\n" + pretty_args)

    # Seed
    setup_seed(args.seed + ddp["rank"])

    # (C) split
    if args.train_json is None or args.val_json is None:
        #
        #
        if is_rank0():
            train_path, valid_path, test_path = prepare_splits(
                source_jsonl=args.source_index,
                out_dir=args.split_dir,
                seed=args.seed,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
            )
            # fill missing CLI args
            if args.train_json is None:
                args.train_json = train_path
            if args.val_json is None:
                args.val_json = valid_path

        # DDP
        # rank0
        if ddp["use_ddp"]:
            obj_list = [args.train_json, args.val_json] if is_rank0() else [None, None]
            dist.broadcast_object_list(obj_list, src=0)
            args.train_json, args.val_json = obj_list[0], obj_list[1]

    # rank0 logging
    if is_rank0():
        print_log(f"Using train_json={args.train_json}")
        print_log(f"Using val_json={args.val_json}")

    # Data
    train_loader, val_loader = make_dataloaders(args, ddp)

    # Model
    model, cfg = build_model(args, device)
    if is_rank0():
        print_log(f"Model params: {count_parameters(model)}")
        print_log(f"Model cfg: {asdict(cfg)}")
        if torch.cuda.is_available():
            print_cuda_memory(device.index if device.type == 'cuda' else 0)

    # DDP wrap
    if ddp["use_ddp"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp["local_rank"]] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.use_cosine:
        # Compute total steps if needed; here epoch-wise schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.cos_min_lr)
    else:
        scheduler = None

    #scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())

    best_rmse = float("inf")
    topk_paths = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        if ddp["use_ddp"]:
            # Shuffle per-epoch for DistributedSampler
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp, args.grad_clip)

        # Scheduler step (epoch)
        if scheduler is not None:
            scheduler.step()

        # Validate
        do_val = (val_loader is not None) and (epoch % args.val_interval == 0 or epoch == args.epochs)
        if do_val and is_rank0():
            val_metrics = validate(model.module if ddp["use_ddp"] else model, val_loader, device)
            cur_rmse = val_metrics["rmse"]
            print_log(
                f"[Epoch {epoch:03d}] "
                f"TrLoss={train_metrics['loss']:7.4f} TrRMSE={train_metrics['rmse']:6.4f} TrMAE={train_metrics['mae']:6.4f} TrR={train_metrics['pearson']:6.4f} | "
                f"VaLoss={val_metrics['loss']:7.4f} VaRMSE={val_metrics['rmse']:6.4f} VaMAE={val_metrics['mae']:6.4f} VaR={val_metrics['pearson']:6.4f} | "
                f"LR={optimizer.param_groups[0]['lr']: .2e}"
            )


            # Save checkpoint if improved
            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                patience_counter = 0  # Reset early stopping counter
                print_log(f"✓ New best Val RMSE: {best_rmse:.4f}")
                state = {
                    "epoch": epoch,
                    "model": (model.module if ddp['use_ddp'] else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "cfg": asdict(cfg),
                    "args": vars(args),
                    "best_rmse": best_rmse,
                }
                tag = f"epoch{epoch:03d}_rmse{best_rmse:.4f}"
                path = save_checkpoint(state, args.save_dir, tag)
                topk_paths.append((best_rmse, path))
                # keep only top-k
                topk_paths = sorted(topk_paths, key=lambda x: x[0])[: max(args.save_topk, 1)]
                # remove extras on disk
                keep_paths = set(p for _, p in topk_paths)
                for f in os.listdir(args.save_dir):
                    if f.startswith("ckpt_epoch") and f.endswith(".pt"):
                        fp = os.path.join(args.save_dir, f)
                        if fp not in keep_paths and args.save_topk > 0:
                            try:
                                os.remove(fp)
                            except Exception:
                                pass
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print_log(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                    break

        elif is_rank0():
            print_log(
                f"[Epoch {epoch:03d}] "
                f"TrLoss={train_metrics['loss']:7.4f} TrRMSE={train_metrics['rmse']:6.4f} TrMAE={train_metrics['mae']:6.4f} TrR={train_metrics['pearson']:6.4f} | "
                f"LR={optimizer.param_groups[0]['lr']: .2e}"
            )

    if is_rank0():
        print_log("Training completed.")
        if torch.cuda.is_available():
            print_cuda_memory(device.index if device.type == 'cuda' else 0)

    cleanup_distributed(ddp["use_ddp"])


if __name__ == "__main__":
    main()
'''
Path("/mnt/data/training.py").write_text(code, encoding="utf-8")
print("Wrote training.py:", Path("/mnt/data/training.py").exists(), "bytes:", len(code))
'''
