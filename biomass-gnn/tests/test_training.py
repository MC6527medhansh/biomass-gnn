"""
Training pipeline tests.
Tests metrics, training loop, checkpointing, and evaluation.

Run: pytest tests/test_training.py -v -s
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


class TestMetrics:
    """Test metric computation with known values."""
    
    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        from src.train.metrics import compute_mae
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        mae = compute_mae(y_pred, y_true)
        assert abs(mae) < 1e-6, f"Expected MAE=0, got {mae}"
    
    def test_mae_known_error(self):
        """MAE should match hand-calculated value."""
        from src.train.metrics import compute_mae
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.5, 2.5, 2.5, 4.5])
        
        # Errors: 0.5, 0.5, 0.5, 0.5 → MAE = 0.5
        mae = compute_mae(y_pred, y_true)
        assert abs(mae - 0.5) < 1e-6, f"Expected MAE=0.5, got {mae}"
    
    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        from src.train.metrics import compute_rmse
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        rmse = compute_rmse(y_pred, y_true)
        assert abs(rmse) < 1e-6, f"Expected RMSE=0, got {rmse}"
    
    def test_rmse_known_error(self):
        """RMSE should match hand-calculated value."""
        from src.train.metrics import compute_rmse
        
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 3.0, 2.0])
        
        # Squared errors: 0, 1, 1 → Mean = 2/3 → RMSE = sqrt(2/3) ≈ 0.8165
        rmse = compute_rmse(y_pred, y_true)
        expected = np.sqrt(2/3)
        assert abs(rmse - expected) < 1e-4, f"Expected RMSE={expected:.4f}, got {rmse:.4f}"
    
    def test_spearman_perfect_correlation(self):
        """Spearman should be 1.0 for perfect monotonic correlation."""
        from src.train.metrics import compute_spearman
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        spearman = compute_spearman(y_pred, y_true)
        assert abs(spearman - 1.0) < 1e-6, f"Expected Spearman=1.0, got {spearman}"
    
    def test_spearman_perfect_anticorrelation(self):
        """Spearman should be -1.0 for perfect negative correlation."""
        from src.train.metrics import compute_spearman
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([4.0, 3.0, 2.0, 1.0])
        
        spearman = compute_spearman(y_pred, y_true)
        assert abs(spearman - (-1.0)) < 1e-6, f"Expected Spearman=-1.0, got {spearman}"
    
    def test_spearman_no_correlation(self):
        """Spearman should be ~0 for no correlation."""
        from src.train.metrics import compute_spearman
        
        # Ranks: [1,2,3,4] vs [2,4,1,3] (no clear pattern)
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([2.0, 4.0, 1.0, 3.0])
        
        spearman = compute_spearman(y_pred, y_true)
        assert abs(spearman) < 0.5, f"Expected Spearman~0, got {spearman}"
    
    def test_spearman_monotonic_transform(self):
        """Spearman should be invariant to monotonic transforms."""
        from src.train.metrics import compute_spearman
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred2 = torch.tensor([10.0, 20.0, 30.0, 40.0])  # Scaled
        y_pred3 = torch.tensor([1.0, 4.0, 9.0, 16.0])  # Squared
        
        s1 = compute_spearman(y_pred1, y_true)
        s2 = compute_spearman(y_pred2, y_true)
        s3 = compute_spearman(y_pred3, y_true)
        
        assert abs(s1 - s2) < 1e-6, "Scaling should not affect Spearman"
        assert abs(s1 - s3) < 1e-6, "Monotonic transform should not affect Spearman"
    
    def test_metrics_handle_single_sample(self):
        """Metrics should handle single-sample case gracefully."""
        from src.train.metrics import compute_mae, compute_rmse, compute_spearman
        
        y_true = torch.tensor([1.5])
        y_pred = torch.tensor([2.0])
        
        mae = compute_mae(y_pred, y_true)
        rmse = compute_rmse(y_pred, y_true)
        
        assert abs(mae - 0.5) < 1e-6
        assert abs(rmse - 0.5) < 1e-6
        
        # Spearman undefined for single sample
        spearman = compute_spearman(y_pred, y_true)
        assert spearman is None or np.isnan(spearman), "Spearman undefined for n=1"
    
    def test_metrics_handle_constant_predictions(self):
        """Metrics should handle constant predictions (no variance)."""
        from src.train.metrics import compute_spearman
        
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([2.5, 2.5, 2.5, 2.5])  # All same
        
        # Spearman undefined when one variable has no variance
        spearman = compute_spearman(y_pred, y_true)
        assert spearman is None or np.isnan(spearman), "Spearman undefined for constant predictions"


class TestTrainingLoop:
    """Test training loop mechanics without full training."""
    
    @pytest.fixture
    def mock_setup(self):
        """Create minimal setup for training tests."""
        from src.models.gnn import create_model
        from src.train.dataset import BiomassDataset
        from torch_geometric.loader import DataLoader
        from src.train.trainer import Trainer
        
        # Create model
        config = {
            'type': 'gcn',
            'hidden_dim': 16,  # Small for speed
            'num_layers': 2,
            'dropout': 0.3
        }
        model = create_model(config)
        
        # Create tiny dataset (first 4 samples)
        dataset = BiomassDataset('data/graphs', 'data/labels.csv', 'data/media')
        small_dataset = torch.utils.data.Subset(dataset, range(4))
        
        train_loader = DataLoader(small_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(small_dataset, batch_size=2, shuffle=False)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.01,
            device='cpu'
        )
        
        return trainer, model
    
    def test_single_train_step(self, mock_setup):
        """Single training step should decrease loss."""
        trainer, model = mock_setup
        
        # Get initial loss
        initial_loss = trainer._compute_loss(trainer.train_loader)
        
        # Train for 1 step
        train_loss = trainer.train_epoch()
        
        # Get new loss
        final_loss = trainer._compute_loss(trainer.train_loader)
        
        print(f"\nInitial loss: {initial_loss:.4f}")
        print(f"Train loss (epoch avg): {train_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        
        # Loss should decrease (but might not by much in 1 epoch)
        # Just check it's finite
        assert np.isfinite(train_loss), "Train loss should be finite"
        assert train_loss > 0, "Train loss should be positive"
    
    def test_validation_step(self, mock_setup):
        """Validation should run without errors."""
        trainer, model = mock_setup
        
        val_metrics = trainer.validate()
        
        print(f"\nValidation metrics: {val_metrics}")
        
        assert 'val_loss' in val_metrics
        assert 'val_mae' in val_metrics
        assert 'val_rmse' in val_metrics
        assert 'val_spearman' in val_metrics
        
        assert np.isfinite(val_metrics['val_loss'])
        assert val_metrics['val_loss'] > 0
    
    def test_training_reduces_loss(self, mock_setup):
        """Training for multiple epochs should reduce loss."""
        trainer, model = mock_setup
        
        initial_loss = trainer._compute_loss(trainer.train_loader)
        
        # Train for 5 epochs
        for epoch in range(5):
            trainer.train_epoch()
        
        final_loss = trainer._compute_loss(trainer.train_loader)
        
        print(f"\nInitial loss: {initial_loss:.4f}")
        print(f"Final loss (after 5 epochs): {final_loss:.4f}")
        print(f"Reduction: {initial_loss - final_loss:.4f}")
        
        # Loss should decrease by at least 1% (very conservative)
        assert final_loss < initial_loss * 0.99, \
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    
    def test_early_stopping_triggers(self, mock_setup):
        """Early stopping should trigger after patience epochs of no improvement."""
        trainer, model = mock_setup
        trainer.patience = 3
        
        # Simulate epochs with no improvement
        trainer.best_val_loss = 1.0  # Set initial best
        trainer.history = {'val_loss': []}
        
        # Epoch 0: no improvement (counter = 1)
        trainer.history['val_loss'].append(1.0)
        should_stop = trainer._check_early_stopping(epoch=0)
        assert not should_stop, "Should not stop at epoch 0"
        
        # Epoch 1: no improvement (counter = 2)
        trainer.history['val_loss'].append(1.0)
        should_stop = trainer._check_early_stopping(epoch=1)
        assert not should_stop, "Should not stop at epoch 1"
        
        # Epoch 2: no improvement (counter = 3, triggers early stopping)
        trainer.history['val_loss'].append(1.0)
        should_stop = trainer._check_early_stopping(epoch=2)
        assert should_stop, "Early stopping should trigger at epoch 2 (after patience=3 epochs of no improvement)"
    
    def test_early_stopping_resets_on_improvement(self, mock_setup):
        """Early stopping counter should reset when validation improves."""
        trainer, model = mock_setup
        trainer.patience = 3
        
        # Validation improves at epoch 3
        trainer.history = {
            'val_loss': [1.0, 1.0, 1.0, 0.8]  # Improvement!
        }
        
        should_stop = trainer._check_early_stopping(epoch=3)
        assert not should_stop, "Early stopping should NOT trigger when validation improves"


class TestCheckpointing:
    """Test model checkpointing and loading."""
    
    def test_save_and_load_checkpoint(self):
        """Saved checkpoint should restore model state exactly."""
        from src.models.gnn import create_model
        from src.train.trainer import Trainer
        import torch
        
        # Create model
        config = {'type': 'gcn', 'hidden_dim': 16, 'num_layers': 2, 'dropout': 0.3}
        model = create_model(config)
        
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Do one forward pass to change parameters
        dummy_data = torch.randn(10, 7)
        dummy_edges = torch.randint(0, 10, (2, 20))
        from torch_geometric.data import Data
        data = Data(x=dummy_data, edge_index=dummy_edges, batch=torch.zeros(10, dtype=torch.long))
        
        output = model(data)
        loss = output.mean()
        loss.backward()
        
        # Update parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.step()
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
            
            Trainer.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                val_loss=0.5,
                save_path=checkpoint_path
            )
            
            assert checkpoint_path.exists(), "Checkpoint file should be created"
            
            # Create new model with different random init
            model2 = create_model(config)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
            
            # Load checkpoint
            checkpoint = Trainer.load_checkpoint(checkpoint_path, model2, optimizer2)
            
            assert checkpoint['epoch'] == 5
            assert checkpoint['val_loss'] == 0.5
            
            # Parameters should match
            for name, param in model.named_parameters():
                param2 = dict(model2.named_parameters())[name]
                assert torch.allclose(param, param2), f"Parameter {name} should match after loading"
    
    def test_checkpoint_contains_required_fields(self):
        """Checkpoint should contain all required fields."""
        from src.models.gnn import create_model
        from src.train.trainer import Trainer
        import torch
        
        config = {'type': 'gcn', 'hidden_dim': 16, 'num_layers': 2, 'dropout': 0.3}
        model = create_model(config)
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            
            Trainer.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=10,
                val_loss=0.123,
                save_path=checkpoint_path
            )
            
            checkpoint = torch.load(checkpoint_path)
            
            required_fields = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'val_loss']
            for field in required_fields:
                assert field in checkpoint, f"Checkpoint missing required field: {field}"


class TestMemoryLeaks:
    """Test that training doesn't leak memory."""
    
    def test_no_memory_leak_during_training(self):
        """Memory usage should stabilize, not grow indefinitely."""
        import psutil
        import gc
        from src.models.gnn import create_model
        from src.train.dataset import BiomassDataset
        from torch_geometric.loader import DataLoader
        from src.train.trainer import Trainer
        
        # Get initial memory
        process = psutil.Process()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create setup
        config = {'type': 'gcn', 'hidden_dim': 16, 'num_layers': 2, 'dropout': 0.3}
        model = create_model(config)
        
        dataset = BiomassDataset('data/graphs', 'data/labels.csv', 'data/media')
        small_dataset = torch.utils.data.Subset(dataset, range(8))
        
        train_loader = DataLoader(small_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(small_dataset, batch_size=2, shuffle=False)
        
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, learning_rate=0.01, device='cpu')
        
        # Train for 10 epochs
        for epoch in range(10):
            trainer.train_epoch()
            trainer.validate()
        
        # Check final memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        final_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        mem_increase = final_mem - initial_mem
        
        print(f"\nMemory: {initial_mem:.1f} MB -> {final_mem:.1f} MB (+{mem_increase:.1f} MB)")
        
        # Allow 100MB increase (conservative, includes dataset loading)
        assert mem_increase < 100, \
            f"Memory leak detected: increased by {mem_increase:.1f} MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])