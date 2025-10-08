"""
Training loop with early stopping and checkpointing.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict
import logging
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles model training, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cpu',
        patience: int = 12
    ):
        """
        Args:
            model: GNN model to train
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            device: 'cpu' or 'cuda'
            patience: Early stopping patience (epochs)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_spearman': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            # Compute loss
            loss = F.mse_loss(output.squeeze(), batch.y.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dict of validation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                output = self.model(batch)
                
                # Compute loss
                loss = F.mse_loss(output.squeeze(), batch.y.squeeze())
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and labels
                all_preds.append(output.squeeze())
                all_labels.append(batch.y.squeeze())
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Compute metrics
        metrics = compute_all_metrics(all_preds, all_labels)
        
        # Add loss
        metrics['val_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Rename keys
        val_metrics = {
            'val_loss': metrics['val_loss'],
            'val_mae': metrics['mae'],
            'val_rmse': metrics['rmse'],
            'val_spearman': metrics['spearman']
        }
        
        return val_metrics
    
    def _compute_loss(self, loader) -> float:
        """
        Compute average loss on a dataset.
        
        Args:
            loader: DataLoader
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = F.mse_loss(output.squeeze(), batch.y.squeeze())
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _check_early_stopping(self, epoch: int) -> bool:
        """
        Check if early stopping criterion is met.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if len(self.history['val_loss']) == 0:
            return False
        
        current_val_loss = self.history['val_loss'][-1]
        
        # Check if validation loss improved (with small tolerance for numerical precision)
        if current_val_loss < self.best_val_loss - 1e-6:
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
            # Save best model state
            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            return False
        else:
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True
            
            return False
    
    def fit(self, num_epochs: int, verbose: bool = True) -> Dict:
        """
        Train model for specified number of epochs with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        logger.info(f"Starting training for up to {num_epochs} epochs")
        logger.info(f"Patience: {self.patience} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_mae'].append(val_metrics['val_mae'])
            self.history['val_rmse'].append(val_metrics['val_rmse'])
            
            # Handle None spearman (can happen with constant predictions)
            spearman = val_metrics['val_spearman']
            self.history['val_spearman'].append(spearman if spearman is not None else 0.0)
            
            # Print progress
            if verbose:
                spearman_str = f"{spearman:.4f}" if spearman is not None else "N/A"
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"MAE: {val_metrics['val_mae']:.4f} | "
                    f"RMSE: {val_metrics['val_rmse']:.4f} | "
                    f"Spearman: {spearman_str}"
                )
            
            # Check early stopping
            if self._check_early_stopping(epoch):
                logger.info(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                if self.best_model_state is not None:
                    self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
                    logger.info("Restored best model from early stopping")
                break
        
        logger.info("Training complete")
        return self.history
    
    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        save_path: Path,
        additional_info: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            val_loss: Validation loss
            save_path: Path to save checkpoint
            additional_info: Optional dict with extra info to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint dict with metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
        
        return checkpoint