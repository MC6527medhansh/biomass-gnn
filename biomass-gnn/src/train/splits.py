"""
Train/Val/Test split logic for biomass dataset.
Uses model-based split to prevent data leakage.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def model_based_split(
    dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset by models to prevent data leakage.
    
    Each model (with all its media conditions) goes entirely into one split.
    This ensures the same graph never appears in multiple splits.
    
    Args:
        dataset: BiomassDataset instance
        val_ratio: Fraction of models for validation (0-1)
        test_ratio: Fraction of models for test (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
        
    Raises:
        ValueError: If ratios are invalid
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1.0")
    
    # Get unique models from dataset samples
    unique_models = list(set([model_id for model_id, _, _ in dataset.samples]))
    n_models = len(unique_models)
    
    if n_models == 0:
        raise ValueError("Dataset has no models")
    
    logger.info(f"Found {n_models} unique models in dataset")
    
    # Shuffle models with fixed seed
    np.random.seed(seed)
    shuffled_models = np.array(unique_models)
    np.random.shuffle(shuffled_models)
    
    # Calculate split sizes (in number of models)
    test_size = int(n_models * test_ratio)
    val_size = int(n_models * val_ratio)
    train_size = n_models - test_size - val_size
    
    if train_size <= 0:
        raise ValueError(f"No models left for training. Adjust val_ratio and test_ratio.")
    
    # Split models into sets
    test_models = set(shuffled_models[:test_size])
    val_models = set(shuffled_models[test_size:test_size + val_size])
    train_models = set(shuffled_models[test_size + val_size:])
    
    logger.info(
        f"Model split: train={train_size} models, "
        f"val={val_size} models, test={test_size} models"
    )
    
    # Verify no overlap
    assert len(train_models & val_models) == 0, "Train-Val model overlap detected"
    assert len(train_models & test_models) == 0, "Train-Test model overlap detected"
    assert len(val_models & test_models) == 0, "Val-Test model overlap detected"
    
    # Assign sample indices based on model membership
    train_idx = []
    val_idx = []
    test_idx = []
    
    for idx, (model_id, media_id, label_idx) in enumerate(dataset.samples):
        if model_id in train_models:
            train_idx.append(idx)  # Use dataset index, not label index
        elif model_id in val_models:
            val_idx.append(idx)
        elif model_id in test_models:
            test_idx.append(idx)
        else:
            logger.warning(f"Sample {idx} with model {model_id} not assigned to any split")
    
    # Log sample statistics
    total_samples = len(train_idx) + len(val_idx) + len(test_idx)
    logger.info(
        f"Sample split: train={len(train_idx)} ({100*len(train_idx)/total_samples:.1f}%), "
        f"val={len(val_idx)} ({100*len(val_idx)/total_samples:.1f}%), "
        f"test={len(test_idx)} ({100*len(test_idx)/total_samples:.1f}%)"
    )
    
    # Verify no sample index overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train-Val sample overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train-Test sample overlap"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val-Test sample overlap"
    
    return train_idx, val_idx, test_idx


def get_split_datasets(dataset, config: dict):
    """
    Create train/val/test splits from a dataset using config parameters.
    
    Uses model-based splitting to prevent data leakage.
    
    Args:
        dataset: BiomassDataset instance
        config: Dict with keys: val_ratio, test_ratio, split_seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import Subset
    
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.20)
    seed = config.get('split_seed', 42)
    
    # Get split indices using model-based split
    train_idx, val_idx, test_idx = model_based_split(
        dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    logger.info(
        f"Created dataset subsets: {len(train_dataset)} train, "
        f"{len(val_dataset)} val, {len(test_dataset)} test"
    )
    
    return train_dataset, val_dataset, test_dataset