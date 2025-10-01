"""
Complete pipeline validation test suite.
Tests everything from data loading through splits.

Run: pytest tests/test_pipeline.py -v
"""
import pytest
import numpy as np
from src.train.dataset import BiomassDataset
from src.train.splits import model_based_split, get_split_datasets


@pytest.fixture
def dataset():
    """Load dataset once for all tests."""
    return BiomassDataset(
        graphs_dir='data/graphs',
        labels_csv='data/labels.csv',
        media_dir='data/media'
    )


class TestDatasetLoading:
    
    def test_dataset_size(self, dataset):
        """Dataset should have 208 samples (212 - 4 human models)."""
        assert len(dataset) == 208, f"Expected 208 samples, got {len(dataset)}"
    
    def test_human_models_filtered(self, dataset):
        """RECON1 and Recon3D should be filtered out."""
        model_ids = set([m for m, _, _ in dataset.samples])
        assert 'RECON1' not in model_ids, "RECON1 should be filtered"
        assert 'Recon3D' not in model_ids, "Recon3D should be filtered"
    
    def test_sample_structure(self, dataset):
        """Each sample should have correct structure."""
        sample = dataset[0]
        
        # Check required attributes
        assert hasattr(sample, 'x'), "Sample missing node features"
        assert hasattr(sample, 'edge_index'), "Sample missing edges"
        assert hasattr(sample, 'edge_attr'), "Sample missing edge features"
        assert hasattr(sample, 'y'), "Sample missing label"
        assert hasattr(sample, 'model_id'), "Sample missing model_id"
        assert hasattr(sample, 'media_id'), "Sample missing media_id"
        
        # Check shapes
        assert sample.x.shape[1] == 7, f"Expected 7 node features, got {sample.x.shape[1]}"
        assert sample.edge_attr.shape[1] == 2, f"Expected 2 edge features, got {sample.edge_attr.shape[1]}"
        assert sample.y.shape == (1,), f"Expected label shape (1,), got {sample.y.shape}"
    
    def test_media_encoding(self, dataset):
        """Media bounds should be encoded on exchange nodes."""
        sample = dataset[0]
        
        # Check media_bound feature exists (7th column)
        media_bounds = sample.x[:, 6]
        
        # At least some nodes should have non-zero bounds
        nonzero = (media_bounds != 0).sum().item()
        assert nonzero > 0, "No media bounds encoded"
        
        # Exchange nodes should have most of the bounds
        is_exchange = sample.x[:, 5]
        exchange_count = (is_exchange == 1).sum().item()
        assert nonzero <= exchange_count, "Non-exchange nodes have media bounds"


class TestSplits:
    
    def test_split_sizes(self, dataset):
        """Splits should have correct proportions."""
        train_idx, val_idx, test_idx = model_based_split(
            dataset, val_ratio=0.15, test_ratio=0.20, seed=42
        )
        
        total = len(train_idx) + len(val_idx) + len(test_idx)
        assert total == len(dataset), f"Split sizes don't sum to dataset size"
        
        # Check approximate ratios
        train_pct = len(train_idx) / total
        val_pct = len(val_idx) / total
        test_pct = len(test_idx) / total
        
        assert 0.60 < train_pct < 0.70, f"Train split {train_pct:.2%} out of range"
        assert 0.10 < val_pct < 0.20, f"Val split {val_pct:.2%} out of range"
        assert 0.15 < test_pct < 0.25, f"Test split {test_pct:.2%} out of range"
    
    def test_no_data_leakage(self, dataset):
        """No model should appear in multiple splits."""
        train_idx, val_idx, test_idx = model_based_split(
            dataset, val_ratio=0.15, test_ratio=0.20, seed=42
        )
        
        # Get models in each split
        train_models = set([dataset.samples[i][0] for i in train_idx])
        val_models = set([dataset.samples[i][0] for i in val_idx])
        test_models = set([dataset.samples[i][0] for i in test_idx])
        
        # Check no overlap
        assert len(train_models & val_models) == 0, "Train-Val model overlap"
        assert len(train_models & test_models) == 0, "Train-Test model overlap"
        assert len(val_models & test_models) == 0, "Val-Test model overlap"
    
    def test_no_index_overlap(self, dataset):
        """No sample index should appear in multiple splits."""
        train_idx, val_idx, test_idx = model_based_split(
            dataset, val_ratio=0.15, test_ratio=0.20, seed=42
        )
        
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        
        assert len(train_set & val_set) == 0, "Train-Val index overlap"
        assert len(train_set & test_set) == 0, "Train-Test index overlap"
        assert len(val_set & test_set) == 0, "Val-Test index overlap"
    
    def test_flux_distribution(self, dataset):
        """All splits should have reasonable flux values."""
        train_idx, val_idx, test_idx = model_based_split(
            dataset, val_ratio=0.15, test_ratio=0.20, seed=42
        )
        
        for split_name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            fluxes = [dataset[i].y.item() for i in indices]
            max_flux = max(fluxes)
            min_flux = min(fluxes)
            
            # Check no outliers (like Recon3D's 755)
            assert max_flux < 10.0, f"{split_name} has outlier flux: {max_flux:.2f}"
            assert min_flux >= 0.0, f"{split_name} has negative flux: {min_flux:.2f}"
    
    def test_media_balance(self, dataset):
        """Each split should have equal minimal and rich samples."""
        train_idx, val_idx, test_idx = model_based_split(
            dataset, val_ratio=0.15, test_ratio=0.20, seed=42
        )
        
        for split_name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            media_ids = [dataset.samples[i][1] for i in indices]
            minimal_count = sum(1 for m in media_ids if m == 'minimal')
            rich_count = sum(1 for m in media_ids if m == 'rich')
            
            assert minimal_count == rich_count, f"{split_name}: {minimal_count} minimal vs {rich_count} rich"
    
    def test_reproducibility(self, dataset):
        """Same seed should give same splits."""
        split1 = model_based_split(dataset, val_ratio=0.15, test_ratio=0.20, seed=42)
        split2 = model_based_split(dataset, val_ratio=0.15, test_ratio=0.20, seed=42)
        
        assert split1[0] == split2[0], "Train splits differ with same seed"
        assert split1[1] == split2[1], "Val splits differ with same seed"
        assert split1[2] == split2[2], "Test splits differ with same seed"


class TestEndToEnd:
    
    def test_full_pipeline(self, dataset):
        """Test complete pipeline from dataset to splits."""
        # Create splits
        config = {
            'val_ratio': 0.15,
            'test_ratio': 0.20,
            'split_seed': 42
        }
        
        train_ds, val_ds, test_ds = get_split_datasets(dataset, config)
        
        # Check sizes
        assert len(train_ds) > 0, "Empty train set"
        assert len(val_ds) > 0, "Empty val set"
        assert len(test_ds) > 0, "Empty test set"
        
        # Load a sample from each
        train_sample = train_ds[0]
        val_sample = val_ds[0]
        test_sample = test_ds[0]
        
        # All should have same structure
        for sample in [train_sample, val_sample, test_sample]:
            assert sample.x.shape[1] == 7
            assert sample.edge_attr.shape[1] == 2
            assert hasattr(sample, 'y')
            assert hasattr(sample, 'model_id')
            assert hasattr(sample, 'media_id')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])