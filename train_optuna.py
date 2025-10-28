# -*- coding: utf-8 -*-
"""
train_optuna.py
- Optuna를 사용한 하이퍼파라미터 튜닝 오케스트레이션
- training.py의 학습 로직을 재사용
- 각 trial마다 optuna_runs/trial_XXX/ 디렉토리 생성

Example:
    python train_optuna.py --n_trials 50 --epochs_per_trial 20 --study_name egnn_hp_tune
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner

import torch
from torch.utils.data import DataLoader

# Existing modules
from utils.logger import print_log, set_log_file
from utils.random_seed import setup_seed
from utils.cuda_utils import print_cuda_memory
from utils.nn_utils import count_parameters, compute_regression_metrics
from utils.split_dataset import prepare_splits
from model.model import BindingAffinityModel, ModelConfig
from model.dataset import PoseDataset, pose_collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for BindingAffinityModel")
    
    # Optuna settings
    p.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    p.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (optional)")
    p.add_argument("--study_name", type=str, default=None, help="Study name (auto-generated if None)")
    p.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Optuna storage URL")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--use_pruner", action="store_true", default=True, help="Use MedianPruner for early stopping")
    p.add_argument("--pruner_n_startup", type=int, default=5, help="Pruner: n_startup_trials")
    p.add_argument("--pruner_n_warmup", type=int, default=10, help="Pruner: n_warmup_steps")
    
    # Data (auto-split if not provided)
    p.add_argument("--source_index", type=str,
                   default="processed/P00533_block/dataset_index.jsonl",
                   help="Master index jsonl for train/val/test splits")
    p.add_argument("--split_dir", type=str, default="index",
                   help="Where to write/read split jsonls")
    p.add_argument("--filter_all_passed", type=int, default=None, choices=[0,1])
    
    # Trial-specific training config
    p.add_argument("--epochs_per_trial", type=int, default=30, help="Max epochs per trial (shorter for HP tuning)")
    p.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience per trial")
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--batch_size_base", type=int, default=16, help="Base batch size (can be tuned by Optuna)")
    
    # Fixed model/training params
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--edge_dim", type=int, default=16)
    p.add_argument("--num_layers_prot", type=int, default=4)
    p.add_argument("--num_layers_lig", type=int, default=4)
    p.add_argument("--num_layers_cross", type=int, default=2)
    p.add_argument("--prot_in_dim", type=int, default=23)
    p.add_argument("--lig_in_dim", type=int, default=771)
    p.add_argument("--use_priors", action="store_true", default=True)
    p.add_argument("--use_cosine", action="store_true", default=True)
    p.add_argument("--cos_min_lr", type=float, default=1e-5)
    p.add_argument("--save_topk", type=int, default=1)  # Only save best model per trial
    
    args = p.parse_args()
    return args


def make_dataloaders(args, train_path, val_path, batch_size):
    """Create dataloaders for a single trial"""
    train_set = PoseDataset(train_path, filter_all_passed=args.filter_all_passed)
    val_set = PoseDataset(val_path, filter_all_passed=args.filter_all_passed)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pose_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pose_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader




def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled, grad_clip):
    model.train()
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
        
        optimizer.zero_grad(set_to_none=True)
        
        if amp_enabled:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                pred = model(**kwargs)
                loss = torch.nn.functional.mse_loss(pred, y)
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
        all_pred.append(pred.detach().cpu().float())
        all_y.append(y.detach().cpu().float())
    
    avg_loss = total_loss / max(total_n, 1)
    
    if len(all_pred) > 0:
        all_pred = torch.cat(all_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)
        train_metrics = compute_regression_metrics(all_pred, all_y)
        train_metrics["loss"] = avg_loss
    else:
        train_metrics = {"loss": avg_loss, "rmse": float("nan"), "mae": float("nan"), "pearson": float("nan"), "r2": float("nan")}
    
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
        return {"loss": float("nan"), "rmse": float("nan"), "mae": float("nan"), "pearson": float("nan"), "r2": float("nan")}
    
    all_pred = torch.cat(all_pred, dim=0)
    all_y = torch.cat(all_y, dim=0)
    metrics = compute_regression_metrics(all_pred, all_y)
    metrics["loss"] = total_loss / total_n
    return metrics


def run_single_trial(args, trial_config: Dict[str, Any], train_loader, val_loader, trial_dir: str, trial: optuna.Trial) -> float:
    """
    Run a single training trial with given hyperparameters.
    
    Returns:
        best_val_rmse (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup seed for reproducibility
    setup_seed(args.seed)
    
    # Build model with trial config
    cfg = ModelConfig(
        hidden_dim=trial_config["hidden_dim"],
        edge_dim=args.edge_dim,
        num_layers_prot=args.num_layers_prot,
        num_layers_lig=args.num_layers_lig,
        num_layers_cross=args.num_layers_cross,
        dropout=trial_config["dropout"],
        prot_in_dim=args.prot_in_dim,
        lig_in_dim=args.lig_in_dim,
        use_priors=args.use_priors,
    )
    
    model = BindingAffinityModel(cfg).to(device)
    
    # Logging setup for this trial
    log_path = os.path.join(trial_dir, "train.log")
    set_log_file(log_path)
    print_log(f"[Trial {trial.number}] Starting training...")
    print_log(f"Trial config: {json.dumps(trial_config, indent=2)}")
    print_log(f"Model params: {count_parameters(model)}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial_config["lr"],
        weight_decay=trial_config["weight_decay"]
    )
    
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_per_trial,
            eta_min=args.cos_min_lr
        )
    else:
        scheduler = None
    
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    
    best_rmse = float("inf")
    patience_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, args.epochs_per_trial + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp, args.grad_clip)
        
        if scheduler is not None:
            scheduler.step()
        
        # Validate every epoch
        val_metrics = validate(model, val_loader, device)
        cur_rmse = val_metrics["rmse"]
        
        # Report to Optuna for pruning
        trial.report(cur_rmse, epoch)
        if trial.should_prune():
            print_log(f"[Trial {trial.number}] Pruned at epoch {epoch}")
            raise optuna.TrialPruned()
        
        print_log(
            f"[Epoch {epoch:03d}] "
            f"TrRMSE={train_metrics['rmse']:6.4f} TrR2={train_metrics['r2']:6.4f} TrP={train_metrics['pearson']:6.4f} | "
            f"VaRMSE={val_metrics['rmse']:6.4f} VaR2={val_metrics['r2']:6.4f} VaP={val_metrics['pearson']:6.4f} | "
            f"LR={optimizer.param_groups[0]['lr']: .2e}"
        )
        
        # Check for improvement
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print_log(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break
    
    print_log(f"[Trial {trial.number}] Completed. Best Val RMSE: {best_rmse:.4f} (epoch {best_epoch})")
    
    return best_rmse


def objective(args, train_loader, val_loader, trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    """
    # Sample hyperparameters
    trial_config = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    
    # Sample batch size (optional, can be fixed)
    batch_size = args.batch_size_base
    
    # Create trial-specific directory
    trial_dir = os.path.join("optuna_runs", f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Run training
    try:
        val_rmse = run_single_trial(args, trial_config, train_loader, val_loader, trial_dir, trial)
        return val_rmse
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print_log(f"[Trial {trial.number}] Error: {e}")
        # Return a bad value but don't raise to continue study
        return float("inf")


def main():
    args = parse_args()
    
    # Auto-generate study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.study_name = f"egnn_tune_{timestamp}"
    
    # Prepare data splits
    print("Preparing data splits...")
    train_path, valid_path, test_path = prepare_splits(
        source_jsonl=args.source_index,
        out_dir=args.split_dir,
        seed=args.seed,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    
    # Create train/val loaders (will be used for all trials)
    print("Creating data loaders...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders(args, train_path, valid_path, batch_size=args.batch_size_base)
    
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Setup Optuna study
    pruner = None
    if args.use_pruner:
        pruner = MedianPruner(
            n_startup_trials=args.pruner_n_startup,
            n_warmup_steps=args.pruner_n_warmup
        )
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",  # We want to minimize val_rmse
        pruner=pruner
    )
    
    print(f"Created/loaded study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Will run {args.n_trials} trials")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(args, train_loader, val_loader, trial),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Optuna Tuning Results")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Val RMSE: {study.best_value:.4f}")
    
    # Save best config with full metrics
    best_config_path = "optuna_runs/best_config.json"
    os.makedirs("optuna_runs", exist_ok=True)
    
    # Get best trial's detailed metrics
    best_trial = study.best_trial
    best_trial_dir = os.path.join("optuna_runs", f"trial_{best_trial.number:04d}")
    best_log_path = os.path.join(best_trial_dir, "train.log")
    
    # Parse the log to get final metrics if available
    # For now, just include basic info
    best_config = {
        "study_name": args.study_name,
        "best_trial_number": study.best_trial.number,
        "best_val_rmse": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_trial_dir": best_trial_dir
    }
    
    with open(best_config_path, "w") as f:
        json.dump(best_config, indent=2)
    
    print(f"\nBest config saved to: {best_config_path}")
    print("\nYou can now use these best params with training.py")


if __name__ == "__main__":
    main()


