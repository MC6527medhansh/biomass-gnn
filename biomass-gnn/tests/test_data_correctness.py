"""
Deep validation tests for data correctness.
Tests that graphs, features, and labels are scientifically accurate.

Run: pytest tests/test_data_correctness.py -v
"""
import pytest
import pandas as pd
import networkx as nx
from src.train.dataset import BiomassDataset
from src.data.sbml_utils import read_sbml_model, find_biomass_reaction, get_bof_stoichiometry


@pytest.fixture
def dataset():
    """Load dataset once."""
    return BiomassDataset(
        graphs_dir='data/graphs',
        labels_csv='data/labels.csv',
        media_dir='data/media'
    )


@pytest.fixture
def labels_df():
    """Load labels CSV."""
    return pd.read_csv('data/labels.csv')


class TestGraphCorrectness:
    
    def test_ecoli_core_structure(self, dataset):
        """Test e_coli_core has correct structure."""
        # Find e_coli_core sample
        ecoli_idx = None
        for idx, (model_id, media_id, _) in enumerate(dataset.samples):
            if model_id == 'e_coli_core' and media_id == 'minimal':
                ecoli_idx = idx
                break
        
        assert ecoli_idx is not None, "e_coli_core not found in dataset"
        
        sample = dataset[ecoli_idx]
        
        # e_coli_core has 72 metabolites + 95 reactions = 167 nodes
        assert sample.x.shape[0] == 167, f"Expected 167 nodes, got {sample.x.shape[0]}"
        
        # Should have 360 edges (from validation earlier)
        assert sample.edge_index.shape[1] == 360, f"Expected 360 edges, got {sample.edge_index.shape[1]}"
        
        # Check biomass reaction has dist_to_biomass = 0
        is_biomass = sample.x[:, 4]
        biomass_nodes = (is_biomass == 1).nonzero().squeeze()
        
        if biomass_nodes.dim() == 0:  # Single node
            biomass_node_idx = biomass_nodes.item()
        else:
            biomass_node_idx = biomass_nodes[0].item()
        
        dist_to_biomass = sample.x[biomass_node_idx, 1].item()
        assert dist_to_biomass == 0, f"Biomass node should have distance 0, got {dist_to_biomass}"
    
    def test_graph_bipartite_structure(self, dataset):
        """Verify graphs are actually bipartite."""
        # Test on first 3 samples
        for idx in [0, 1, 2]:
            sample = dataset[idx]
            
            # Get node types
            node_types = sample.x[:, 0].numpy()
            
            # Check edges connect different node types
            edge_index = sample.edge_index.numpy()
            
            for i in range(min(100, edge_index.shape[1])):  # Check first 100 edges
                src = edge_index[0, i]
                dst = edge_index[1, i]
                
                src_type = node_types[src]
                dst_type = node_types[dst]
                
                # Bipartite: edges should connect different types
                assert src_type != dst_type, f"Sample {idx}: Edge {src}->{dst} connects same types"
    
    def test_bof_features(self, dataset):
        """Verify BOF features are correctly set."""
        # Get e_coli_core sample
        ecoli_idx = None
        for idx, (model_id, media_id, _) in enumerate(dataset.samples):
            if model_id == 'e_coli_core':
                ecoli_idx = idx
                break
        
        sample = dataset[ecoli_idx]
        
        # Load actual BOF from model
        model = read_sbml_model('data/models/e_coli_core.xml', validate=False)
        biomass = find_biomass_reaction(model)
        bof_stoich = get_bof_stoichiometry(model, biomass)
        
        # e_coli_core BOF has 16 metabolites
        assert len(bof_stoich) == 16, f"Expected 16 BOF metabolites, got {len(bof_stoich)}"
        
        # Count nodes with in_bof=1
        in_bof = sample.x[:, 2]
        bof_count = (in_bof == 1).sum().item()
        
        # Should match (or be close if some metabolites missing from graph)
        assert 10 <= bof_count <= 20, f"Expected ~16 BOF nodes, got {bof_count}"


class TestMediaEncoding:
    
    def test_minimal_vs_rich_difference(self, dataset):
        """Verify rich media has more exchanges than minimal."""
        # Find a model with both media
        test_model = 'iJO1366'
        
        minimal_idx = None
        rich_idx = None
        
        for idx, (model_id, media_id, _) in enumerate(dataset.samples):
            if model_id == test_model:
                if media_id == 'minimal':
                    minimal_idx = idx
                elif media_id == 'rich':
                    rich_idx = idx
        
        assert minimal_idx is not None and rich_idx is not None, f"{test_model} not found"
        
        minimal_sample = dataset[minimal_idx]
        rich_sample = dataset[rich_idx]
        
        # Count non-zero media bounds
        minimal_bounds = (minimal_sample.x[:, 6] != 0).sum().item()
        rich_bounds = (rich_sample.x[:, 6] != 0).sum().item()
        
        assert rich_bounds > minimal_bounds, f"Rich ({rich_bounds}) should have more bounds than minimal ({minimal_bounds})"
    
    def test_media_matches_model_medium(self, dataset):
        """Verify encoded media matches model's actual medium."""
        # Test e_coli_core
        ecoli_idx = None
        for idx, (model_id, media_id, _) in enumerate(dataset.samples):
            if model_id == 'e_coli_core' and media_id == 'minimal':
                ecoli_idx = idx
                break
        
        sample = dataset[ecoli_idx]
        
        # Load model's actual medium
        model = read_sbml_model('data/models/e_coli_core.xml', validate=False)
        actual_medium = dict(model.medium)
        
        # e_coli_core has 7 exchanges in default medium
        assert len(actual_medium) == 7, f"Expected 7 exchanges in e_coli_core medium"
        
        # Count encoded non-zero bounds
        media_bounds = (sample.x[:, 6] != 0).sum().item()
        
        # Should match (approximately - some exchanges might not be in graph)
        assert 5 <= media_bounds <= 10, f"Expected ~7 encoded bounds, got {media_bounds}"
    
    def test_exchange_nodes_have_bounds(self, dataset):
        """Media bounds should only be on exchange nodes."""
        sample = dataset[0]
        
        is_exchange = sample.x[:, 5]
        media_bounds = sample.x[:, 6]
        
        # Find nodes with non-zero bounds that aren't exchanges
        non_exchange_with_bounds = ((is_exchange == 0) & (media_bounds != 0)).sum().item()
        
        assert non_exchange_with_bounds == 0, f"{non_exchange_with_bounds} non-exchange nodes have media bounds"


class TestLabelAlignment:
    
    def test_label_matches_csv(self, dataset, labels_df):
        """Verify dataset labels match labels.csv."""
        # Test first 10 samples
        for idx in range(min(10, len(dataset))):
            sample = dataset[idx]
            model_id, media_id, label_idx = dataset.samples[idx]
            
            # Get label from dataset
            dataset_flux = sample.y.item()
            
            # Get label from CSV
            csv_row = labels_df.iloc[label_idx]
            csv_flux = csv_row['biomass_flux']
            
            # They should match
            assert abs(dataset_flux - csv_flux) < 0.0001, \
                f"Sample {idx} ({model_id}|{media_id}): dataset={dataset_flux:.4f}, csv={csv_flux:.4f}"
    
    def test_grow_flag_consistency(self, dataset, labels_df):
        """Verify grow flag matches flux values."""
        for idx in range(min(20, len(dataset))):
            sample = dataset[idx]
            _, _, label_idx = dataset.samples[idx]
            
            flux = sample.y.item()
            grow = sample.grow
            
            # grow should be 1 if flux > 0, else 0
            expected_grow = 1 if flux > 1e-6 else 0
            
            assert grow == expected_grow, \
                f"Sample {idx}: flux={flux:.4f}, grow={grow}, expected={expected_grow}"
    
    def test_no_human_model_labels(self, dataset, labels_df):
        """Verify no samples use RECON1/Recon3D labels."""
        for idx in range(len(dataset)):
            model_id, _, label_idx = dataset.samples[idx]
            
            # Check the label isn't from a human model
            csv_model = labels_df.iloc[label_idx]['model_id']
            
            assert csv_model not in ['RECON1', 'Recon3D'], \
                f"Sample {idx} uses label from filtered model {csv_model}"
                
    def test_random_sample_correctness(self, dataset, labels_df):
        """Test 20 random models for correctness."""
        import random
        random.seed(42)
        
        # Get unique models
        unique_models = list(set([m for m, _, _ in dataset.samples]))
        test_models = random.sample(unique_models, min(20, len(unique_models)))
        
        for model_id in test_models:
            # Find sample for this model
            idx = next(i for i, (m, _, _) in enumerate(dataset.samples) if m == model_id)
            sample = dataset[idx]
            
            # Basic checks
            assert sample.x.shape[0] > 0, f"{model_id}: No nodes"
            assert sample.edge_index.shape[1] > 0, f"{model_id}: No edges"
            assert sample.y.item() >= 0, f"{model_id}: Negative flux"
            
            # Check bipartite
            node_types = sample.x[:, 0]
            assert ((node_types == 0) | (node_types == 1)).all(), f"{model_id}: Invalid node types"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])