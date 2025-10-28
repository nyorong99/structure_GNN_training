# -*- coding: utf-8 -*-
"""
nn_utils.py
- Neural network related utility functions.
"""
import torch
from typing import Dict

def count_parameters(model) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))

def compute_regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute regression metrics: RMSE, MAE, Pearson correlation, R2.
    
    Args:
        pred: Predictions tensor (N,) or (N, 1)
        target: Ground truth tensor (N,) or (N, 1)
    
    Returns:
        Dictionary with keys: 'rmse', 'mae', 'pearson', 'r2'
    """
    # Ensure 1D
    if target.ndim == 2 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if pred.ndim == 2 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    
    with torch.no_grad():
        diff = pred - target
        
        # MSE and RMSE
        mse = torch.mean(diff * diff)
        rmse = torch.sqrt(mse + 1e-12)
        
        # MAE
        mae = torch.mean(torch.abs(diff))
        
        # Pearson correlation
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        num = torch.sum(pred_centered * target_centered)
        den = torch.sqrt(torch.sum(pred_centered * pred_centered) * torch.sum(target_centered * target_centered) + 1e-12)
        pearson = num / den if den > 0 else torch.tensor(0.0, device=pred.device)
        
        # R2 score
        ss_res = torch.sum(diff * diff)
        ss_tot = torch.sum(target_centered * target_centered)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    
    return {
        "rmse": rmse.item(),
        "mae": mae.item(),
        "pearson": pearson.item(),
        "r2": r2.item(),
    }
