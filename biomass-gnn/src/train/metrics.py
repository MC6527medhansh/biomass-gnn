"""
Metrics for regression tasks.
Computes MAE, RMSE, and Spearman correlation.
"""
import torch
import numpy as np
from scipy.stats import spearmanr
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_pred: Predictions [N] or [N, 1]
        y_true: True values [N] or [N, 1]
        
    Returns:
        MAE as float
    """
    y_pred = y_pred.detach().cpu().flatten()
    y_true = y_true.detach().cpu().flatten()
    
    mae = torch.abs(y_pred - y_true).mean().item()
    return mae


def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_pred: Predictions [N] or [N, 1]
        y_true: True values [N] or [N, 1]
        
    Returns:
        RMSE as float
    """
    y_pred = y_pred.detach().cpu().flatten()
    y_true = y_true.detach().cpu().flatten()
    
    mse = torch.pow(y_pred - y_true, 2).mean().item()
    rmse = np.sqrt(mse)
    return rmse


def compute_spearman(y_pred: torch.Tensor, y_true: torch.Tensor) -> Optional[float]:
    """
    Compute Spearman rank correlation coefficient.
    
    Measures monotonic relationship between predictions and true values.
    Range: [-1, 1] where 1 = perfect positive correlation, -1 = perfect negative.
    
    Args:
        y_pred: Predictions [N] or [N, 1]
        y_true: True values [N] or [N, 1]
        
    Returns:
        Spearman correlation as float, or None if undefined
    """
    y_pred = y_pred.detach().cpu().flatten().numpy()
    y_true = y_true.detach().cpu().flatten().numpy()
    
    # Spearman undefined for n < 2
    if len(y_pred) < 2:
        return None
    
    # Spearman undefined if either variable has no variance
    if np.var(y_pred) < 1e-10 or np.var(y_true) < 1e-10:
        return None
    
    try:
        corr, _ = spearmanr(y_pred, y_true)
        
        # Handle NaN (can occur with ties or numerical issues)
        if np.isnan(corr):
            return None
        
        return float(corr)
    
    except Exception as e:
        logger.warning(f"Spearman correlation failed: {e}")
        return None


def compute_all_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    Compute all regression metrics.
    
    Args:
        y_pred: Predictions [N] or [N, 1]
        y_true: True values [N] or [N, 1]
        
    Returns:
        Dict with keys: mae, rmse, spearman
    """
    metrics = {
        'mae': compute_mae(y_pred, y_true),
        'rmse': compute_rmse(y_pred, y_true),
        'spearman': compute_spearman(y_pred, y_true)
    }
    
    return metrics