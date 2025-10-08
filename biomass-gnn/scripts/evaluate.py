"""
Evaluation script for trained models.
Creates visualizations and detailed analysis.

Usage:
    python scripts/evaluate.py --checkpoint runs/experiment1/best_model.pt --config configs/base.yaml
"""
import argparse
import yaml
import torch
from pathlib import Path
import logging
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn import create_model
from src.train.dataset import BiomassDataset
from src.train.splits import get_split_datasets
from src.train.trainer import Trainer
from src.train.metrics import compute_all_metrics
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model, loader, device='cpu'):
    """
    Evaluate model on a dataset.
    
    Returns:
        predictions, labels, model_ids, media_ids
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_model_ids = []
    all_media_ids = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            
            all_preds.append(output.squeeze().cpu())
            all_labels.append(batch.y.squeeze().cpu())
            all_model_ids.extend(batch.model_id)
            all_media_ids.extend(batch.media_id)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return all_preds, all_labels, all_model_ids, all_media_ids


def plot_predictions(y_pred, y_true, output_path, title="Predictions vs True"):
    """Create scatter plot of predictions vs true values."""
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Biomass Flux', fontsize=12)
    plt.ylabel('Predicted Biomass Flux', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def plot_residuals(y_pred, y_true, output_path):
    """Create residual plot."""
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs predictions
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=50)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Biomass Flux', fontsize=12)
    axes[0].set_ylabel('Residual (Pred - True)', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved residual plot to {output_path}")


def analyze_by_media(y_pred, y_true, media_ids, output_path):
    """Analyze performance by media condition."""
    media_groups = defaultdict(lambda: {'pred': [], 'true': []})
    
    for pred, true, media in zip(y_pred, y_true, media_ids):
        media_groups[media]['pred'].append(pred)
        media_groups[media]['true'].append(true)
    
    results = {}
    for media, data in media_groups.items():
        pred = torch.tensor(data['pred'])
        true = torch.tensor(data['true'])
        metrics = compute_all_metrics(pred, true)
        results[media] = metrics
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Media-specific results saved to {output_path}")
    
    # Print summary
    print("\nPerformance by Media Condition:")
    for media, metrics in results.items():
        spearman_str = f"{metrics['spearman']:.4f}" if metrics['spearman'] is not None else "N/A"
        print(f"  {media:8s} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | Spearman: {spearman_str}")


def analyze_by_organism(y_pred, y_true, model_ids, output_path, top_n=10):
    """Analyze best and worst performing organisms."""
    organism_groups = defaultdict(lambda: {'pred': [], 'true': []})
    
    for pred, true, model_id in zip(y_pred, y_true, model_ids):
        organism_groups[model_id]['pred'].append(pred)
        organism_groups[model_id]['true'].append(true)
    
    organism_mae = {}
    for model_id, data in organism_groups.items():
        pred = torch.tensor(data['pred'])
        true = torch.tensor(data['true'])
        mae = compute_all_metrics(pred, true)['mae']
        organism_mae[model_id] = mae
    
    # Sort by MAE
    sorted_organisms = sorted(organism_mae.items(), key=lambda x: x[1])
    
    # Best and worst
    best = sorted_organisms[:top_n]
    worst = sorted_organisms[-top_n:]
    
    results = {
        'best_organisms': [{'model_id': m, 'mae': mae} for m, mae in best],
        'worst_organisms': [{'model_id': m, 'mae': mae} for m, mae in worst]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Organism analysis saved to {output_path}")
    
    # Print summary
    print(f"\nTop {top_n} Best Predicted Organisms (Lowest MAE):")
    for model_id, mae in best:
        print(f"  {model_id:20s} | MAE: {mae:.4f}")
    
    print(f"\nTop {top_n} Worst Predicted Organisms (Highest MAE):")
    for model_id, mae in worst:
        print(f"  {model_id:20s} | MAE: {mae:.4f}")


def plot_training_curves(history_path, output_path):
    """Plot training and validation curves."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history['val_mae'], label='Val MAE', marker='s', markersize=3, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(epochs, history['val_rmse'], label='Val RMSE', marker='s', markersize=3, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spearman
    axes[1, 1].plot(epochs, history['val_spearman'], label='Val Spearman', marker='s', markersize=3, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Spearman Correlation')
    axes[1, 1].set_title('Validation Spearman Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: same as checkpoint dir)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine output directory
    checkpoint_path = Path(args.checkpoint)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = checkpoint_path.parent / 'evaluation'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = BiomassDataset(
        graphs_dir=config['data']['graphs_dir'],
        labels_csv=config['data']['labels_csv'],
        media_dir='data/media'
    )
    
    # Create splits
    train_dataset, val_dataset, test_dataset = get_split_datasets(dataset, config['data'])
    
    # Create loaders
    batch_size = config['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = create_model(config['model'])
    Trainer.load_checkpoint(checkpoint_path, model)
    model = model.to(args.device)
    
    # Evaluate on all splits
    logger.info("\nEvaluating on all splits...")
    
    results = {}
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        logger.info(f"Evaluating {split_name} set...")
        
        y_pred, y_true, model_ids, media_ids = evaluate_model(model, loader, args.device)
        
        # Compute metrics
        metrics = compute_all_metrics(torch.tensor(y_pred), torch.tensor(y_true))
        
        results[split_name] = {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'spearman': metrics['spearman'],
            'n_samples': len(y_pred)
        }
        
        # Create plots for test set
        if split_name == 'test':
            plot_predictions(
                y_pred, y_true,
                output_dir / 'test_predictions.png',
                title=f"Test Set Predictions (n={len(y_pred)})"
            )
            
            plot_residuals(
                y_pred, y_true,
                output_dir / 'test_residuals.png'
            )
            
            analyze_by_media(
                y_pred, y_true, media_ids,
                output_dir / 'test_by_media.json'
            )
            
            analyze_by_organism(
                y_pred, y_true, model_ids,
                output_dir / 'test_by_organism.json'
            )
    
    # Save overall results
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for split_name, split_results in results.items():
        spearman_str = f"{split_results['spearman']:.4f}" if split_results['spearman'] is not None else "N/A"
        print(f"\n{split_name.upper()} SET (n={split_results['n_samples']}):")
        print(f"  MAE:      {split_results['mae']:.4f}")
        print(f"  RMSE:     {split_results['rmse']:.4f}")
        print(f"  Spearman: {spearman_str}")
    
    # Plot training curves if history available
    history_path = checkpoint_path.parent / 'history.json'
    if history_path.exists():
        plot_training_curves(history_path, output_dir / 'training_curves.png')
    
    print("\n" + "="*60)
    print(f"Evaluation complete! Results saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()