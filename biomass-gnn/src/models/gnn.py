"""
Graph Neural Network models for biomass prediction.
Supports GCN, GAT, and GraphSAGE architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import logging

logger = logging.getLogger(__name__)


class BiomassGNN(nn.Module):
    """
    Graph Neural Network for predicting biomass flux from metabolic networks.
    
    Architecture:
    - Initial projection: input_dim → hidden_dim
    - N GNN layers with dropout
    - Global pooling
    - MLP head for regression
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        gnn_type: str = 'gcn'
    ):
        """
        Args:
            input_dim: Number of node features (default: 7)
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN ('gcn', 'gat', 'sage')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        
        # Initial projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if self.gnn_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif self.gnn_type == 'gat':
                conv = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
            elif self.gnn_type == 'sage':
                conv = SAGEConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # MLP head for regression
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(
            f"Initialized {gnn_type.upper()} with {num_layers} layers, "
            f"hidden_dim={hidden_dim}, dropout={dropout}"
        )
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object with attributes:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            Predicted biomass flux [batch_size, 1]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Apply GNN
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection (if not first layer)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP head
        out = self.mlp(x)
        
        return out
    
    def get_embeddings(self, data):
        """
        Get node embeddings after GNN layers (for analysis).
        
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            
            if i > 0:
                x = x + x_new
            else:
                x = x_new
        
        return x


def create_model(config: dict) -> BiomassGNN:
    """
    Create GNN model from config dictionary.
    
    Args:
        config: Dict with keys:
            - type: 'gcn', 'gat', or 'sage'
            - hidden_dim: int
            - num_layers: int
            - dropout: float
            
    Returns:
        Initialized BiomassGNN model
    """
    model = BiomassGNN(
        input_dim=7,  # Fixed: 7 node features
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        gnn_type=config.get('type', 'gcn')
    )
    
    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} parameters ({num_trainable:,} trainable)")
    
    return model