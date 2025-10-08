"""
Training script for biomass prediction GNN.

Usage:
    python scripts/train.py --config configs/base.yaml --output runs/experiment1
"""
import argparse
import yaml
import torch
from pathlib import Path
import logging
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn import create_model
from src.train.dataset import BiomassDataset
from src.train.splits import get_split_datasets
from src.train.trainer import Trainer
from torch_geometric.loader import DataLoader
from src.utils.seed import set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train biomass prediction GNN')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='runs/default',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed if args.seed is not None else config['data'].get('split_seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config to output dir
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = BiomassDataset(
        graphs_dir=config['data']['graphs_dir'],
        labels_csv=config['data']['labels_csv'],
        media_dir='data/media'
    )
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Create splits
    logger.info("Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = get_split_datasets(dataset, config['data'])
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    logger.info(f"  Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    batch_size = config['train']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config['model'])
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config['model']['type'].upper()}")
    logger.info(f"Parameters: {num_params:,}")
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
        device=args.device,
        patience=config['train']['early_stop_patience']
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Max epochs: {config['train']['epochs']}")
    logger.info(f"Early stopping patience: {config['train']['early_stop_patience']}")
    
    history = trainer.fit(
        num_epochs=config['train']['epochs'],
        verbose=True
    )
    
    # Save final model
    checkpoint_path = output_dir / 'final_model.pt'
    Trainer.save_checkpoint(
        model=model,
        optimizer=trainer.optimizer,
        epoch=len(history['train_loss']),
        val_loss=history['val_loss'][-1],
        save_path=checkpoint_path,
        additional_info={'history': history}
    )
    
    # Save best model (from early stopping)
    if trainer.best_model_state is not None:
        best_checkpoint_path = output_dir / 'best_model.pt'
        best_model = create_model(config['model'])
        best_model.load_state_dict(trainer.best_model_state)
        Trainer.save_checkpoint(
            model=best_model,
            optimizer=trainer.optimizer,
            epoch=len(history['train_loss']) - trainer.epochs_without_improvement,
            val_loss=trainer.best_val_loss,
            save_path=best_checkpoint_path
        )
        logger.info(f"Saved best model to {best_checkpoint_path}")
    
    # Save training history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        # Convert None values to "null" for JSON
        history_clean = {k: [v if v is not None else None for v in vals] 
                        for k, vals in history.items()}
        json.dump(history_clean, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Final evaluation on test set
    logger.info("\n" + "="*60)
    logger.info("Evaluating on test set...")
    
    # Load best model
    if trainer.best_model_state is not None:
        model.load_state_dict({k: v.to(args.device) for k, v in trainer.best_model_state.items()})
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(args.device)
            output = model(batch)
            all_preds.append(output.squeeze())
            all_labels.append(batch.y.squeeze())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute test metrics
    from src.train.metrics import compute_all_metrics
    test_metrics = compute_all_metrics(all_preds, all_labels)
    
    logger.info("Test Set Results:")
    logger.info(f"  MAE:      {test_metrics['mae']:.4f}")
    logger.info(f"  RMSE:     {test_metrics['rmse']:.4f}")
    logger.info(f"  Spearman: {test_metrics['spearman']:.4f}" if test_metrics['spearman'] is not None else "  Spearman: N/A")
    
    # Save test results
    test_results = {
        'mae': test_metrics['mae'],
        'rmse': test_metrics['rmse'],
        'spearman': test_metrics['spearman']
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("="*60)
    logger.info(f"Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()