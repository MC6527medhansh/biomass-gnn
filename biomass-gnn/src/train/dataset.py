"""
PyTorch Geometric Dataset for biomass prediction.
Loads graphs from GraphML and encodes media conditions on exchange reaction nodes.

FIXED: Media encoding now correctly uses CSV files instead of model.medium
"""
import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class BiomassDataset(Dataset):
    """
    Dataset for graph-level biomass prediction with media conditioning.
    
    Each sample is a (metabolic_graph, media_condition) pair with a biomass_flux label.
    Media conditions are encoded as node features on exchange reactions.
    """
    
    def __init__(
        self,
        graphs_dir: str,
        labels_csv: str,
        media_dir: str,
        transform=None,
        pre_transform=None
    ):
        """
        Args:
            graphs_dir: Directory with .graphml files
            labels_csv: Path to labels CSV (model_id, media_id, biomass_flux, grow)
            media_dir: Directory with media CSV files
            transform: Optional transform to apply on-the-fly
            pre_transform: Optional transform to apply once during processing
        """
        self.graphs_dir = graphs_dir
        self.labels_csv = labels_csv
        self.media_dir = media_dir
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        logger.info(f"Loaded {len(self.labels_df)} labels from {labels_csv}")
        
        # Load media profiles
        self.media_profiles = self._load_media_profiles()
        logger.info(f"Loaded {len(self.media_profiles)} media profiles")
        
        # Build sample list: (model_id, media_id) pairs
        self.samples = self._build_sample_list()
        logger.info(f"Created {len(self.samples)} samples")
        
        # Cache for loaded graphs (avoid reloading same GraphML)
        self._graph_cache = {}
        
        super().__init__(None, transform, pre_transform)
    
    def _load_media_profiles(self) -> Dict[str, Dict[str, float]]:
        """Load all media CSV files into dict of {media_name: {rxn_id: bound}}."""
        from src.data.media import load_media_dir
        media_list = load_media_dir(self.media_dir)
        return {m.name: m.bounds for m in media_list}
    
    def _build_sample_list(self) -> List[tuple]:
        """
        Build list of (model_id, media_id, label_idx) tuples.
        Only include samples where GraphML file exists.
        Filters out human metabolic models (different units/objectives).
        """
        samples = []
        missing_graphs = set()
        filtered_models = set()
        
        # Human models to exclude (different biomass definitions)
        human_models = {'RECON1', 'Recon3D'}
        
        for idx, row in self.labels_df.iterrows():
            model_id = row['model_id']
            media_id = row['media_id']
            
            # Filter human models
            if model_id in human_models:
                filtered_models.add(model_id)
                continue
            
            graph_path = os.path.join(self.graphs_dir, f"{model_id}.graphml")
            
            if not os.path.exists(graph_path):
                missing_graphs.add(model_id)
                continue
            
            samples.append((model_id, media_id, idx))
        
        if missing_graphs:
            logger.warning(
                f"Missing graphs for {len(missing_graphs)} models: "
                f"{list(missing_graphs)[:5]}..."
            )
        
        if filtered_models:
            logger.info(
                f"Filtered {len(filtered_models)} human models: {filtered_models}"
            )
        
        return samples
    
    def len(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def get(self, idx: int) -> Data:
        """
        Load and return a single sample.
        
        Returns:
            PyG Data object with:
                - x: node features [num_nodes, num_features]
                - edge_index: graph connectivity [2, num_edges]
                - edge_attr: edge features [num_edges, num_edge_features]
                - y: biomass flux label (scalar)
                - model_id: string identifier
                - media_id: string identifier
        """
        model_id, media_id, label_idx = self.samples[idx]
        
        # Get graph path
        graph_path = os.path.join(self.graphs_dir, f"{model_id}.graphml")
        
        # Load graph (with caching)
        if model_id not in self._graph_cache:
            G = nx.read_graphml(graph_path)
            self._graph_cache[model_id] = G
        else:
            G = self._graph_cache[model_id].copy()
        
        # Encode media on exchange reactions
        G = self._encode_media(G, media_id)
        
        # Convert to PyG Data
        data = self._networkx_to_pyg(G)
        
        # Add label
        label_row = self.labels_df.iloc[label_idx]
        data.y = torch.tensor([label_row['biomass_flux']], dtype=torch.float)
        
        # Add metadata
        data.model_id = model_id
        data.media_id = media_id
        data.grow = int(label_row['grow'])
        
        return data
    
    def _encode_media(self, G: nx.Graph, media_id: str) -> nx.Graph:
        """
        Add media_bound feature to exchange reaction nodes using CSV media profiles.
        
        FIXED: Now uses self.media_profiles (loaded from CSV files) instead of model.medium.
        
        For each exchange reaction node:
        - If rxn_id in media profile CSV: set media_bound = bound value (negative = uptake)
        - Otherwise: set media_bound = 0.0 (closed/no uptake)
        
        Non-exchange nodes always get media_bound = 0.0
        
        Args:
            G: NetworkX graph with node attributes
            media_id: Media condition name (e.g., 'minimal', 'rich')
            
        Returns:
            Graph with media_bound feature added to all nodes
            
        Raises:
            ValueError: If media_id not found in loaded media profiles
        """
        # Validate media_id
        if media_id not in self.media_profiles:
            available = list(self.media_profiles.keys())
            raise ValueError(
                f"Unknown media condition: '{media_id}'. "
                f"Available: {available}"
            )
        
        # Get media bounds from CSV (not from model!)
        media_bounds = self.media_profiles[media_id]
        
        # Encode on graph nodes
        matched_count = 0
        total_exchanges = 0
        
        for node in G.nodes():
            node_data = G.nodes[node]
            is_exchange = int(node_data.get('is_exchange', 0))
            
            if is_exchange:
                total_exchanges += 1
                
                # Check if this exchange is in the media profile
                if node in media_bounds:
                    G.nodes[node]['media_bound'] = float(media_bounds[node])
                    matched_count += 1
                else:
                    # Exchange not in media = closed (no uptake)
                    G.nodes[node]['media_bound'] = 0.0
            else:
                # Non-exchange nodes always get 0
                G.nodes[node]['media_bound'] = 0.0
        
        # Calculate match rate
        match_rate = 100 * matched_count / total_exchanges if total_exchanges > 0 else 0
        
        logger.debug(
            f"Encoded media '{media_id}': matched {matched_count}/{total_exchanges} "
            f"exchanges ({match_rate:.1f}%)"
        )
        
        # Warn if match rate is suspiciously low
        if match_rate < 5 and total_exchanges > 0:
            logger.warning(
                f"Low media match rate ({match_rate:.1f}%) for media '{media_id}'. "
                f"Check that exchange reaction IDs in CSV match graph node IDs."
            )
        
        return G
    
    def _networkx_to_pyg(self, G: nx.Graph) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Manually constructs tensors to handle heterogeneous node attributes.
        
        Node features (7 dimensions):
        - node_type (0=metabolite, 1=reaction)
        - dist_to_biomass
        - in_bof
        - bof_coef
        - is_biomass
        - is_exchange
        - media_bound
        
        Edge features (2 dimensions):
        - stoich_coef
        - is_reactant
        """
        # Create node ID mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Build node feature matrix
        num_nodes = len(node_list)
        node_features = []
        
        for node in node_list:
            node_data = G.nodes[node]
            
            # Extract features with defaults for missing values
            features = [
                float(node_data.get('node_type', 0)),
                float(node_data.get('dist_to_biomass', -1)),
                float(node_data.get('in_bof', 0)),
                float(node_data.get('bof_coef', 0.0)),
                float(node_data.get('is_biomass', 0)),
                float(node_data.get('is_exchange', 0)),
                float(node_data.get('media_bound', 0.0))
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Build edge index and edge attributes
        edge_list = []
        edge_features = []
        
        for src, dst, edge_data in G.edges(data=True):
            src_idx = node_to_idx[src]
            dst_idx = node_to_idx[dst]
            
            edge_list.append([src_idx, dst_idx])
            
            # Extract edge features
            edge_features.append([
                float(edge_data.get('stoich_coef', 1.0)),
                float(edge_data.get('is_reactant', 0))
            ])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data