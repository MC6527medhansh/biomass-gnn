"""
Feature computation for metabolic network graphs.
Adds biomass-aware features to graph nodes.
"""
import networkx as nx
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_distance_to_biomass(G: nx.DiGraph, biomass_rxn_id: str) -> Dict[str, int]:
    """
    Compute shortest path distance from each node to the biomass reaction.
    
    Uses reverse BFS from biomass reaction, following edges backwards.
    This gives the distance in the direction of metabolic flow toward biomass.
    
    Args:
        G: NetworkX DiGraph (directed bipartite graph)
        biomass_rxn_id: ID of biomass reaction node
        
    Returns:
        Dict mapping node_id -> distance (int)
        Nodes unreachable from biomass get distance = -1
        
    Raises:
        ValueError: If biomass_rxn_id not in graph
    """
    if biomass_rxn_id not in G:
        raise ValueError(f"Biomass reaction '{biomass_rxn_id}' not found in graph")
    
    # Reverse the graph to trace backwards from biomass
    G_reversed = G.reverse(copy=True)
    
    # BFS from biomass in reversed graph
    distances = {}
    visited = {biomass_rxn_id}
    queue = [(biomass_rxn_id, 0)]
    
    while queue:
        node, dist = queue.pop(0)
        distances[node] = dist
        
        # Visit neighbors
        for neighbor in G_reversed.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    # Set unreachable nodes to -1
    for node in G.nodes:
        if node not in distances:
            distances[node] = -1
    
    reachable = sum(1 for d in distances.values() if d >= 0)
    unreachable = sum(1 for d in distances.values() if d == -1)
    
    logger.info(
        f"Distance computation: {reachable} reachable nodes, "
        f"{unreachable} unreachable from biomass"
    )
    
    if unreachable > 0:
        logger.warning(
            f"{unreachable}/{len(distances)} nodes unreachable from biomass "
            f"(disconnected components)"
        )
    
    return distances


def add_distance_features(G: nx.DiGraph, biomass_rxn_id: str) -> nx.DiGraph:
    """
    Add dist_to_biomass attribute to all nodes in graph.
    
    Modifies graph in-place and returns it.
    
    Args:
        G: NetworkX DiGraph
        biomass_rxn_id: ID of biomass reaction
        
    Returns:
        Modified graph with dist_to_biomass attribute on all nodes
    """
    distances = compute_distance_to_biomass(G, biomass_rxn_id)
    
    # Add as node attribute
    for node, dist in distances.items():
        G.nodes[node]['dist_to_biomass'] = dist
    
    logger.debug(f"Added dist_to_biomass feature to {len(distances)} nodes")
    
    return G


def add_bof_features(
    G: nx.DiGraph, 
    bof_stoichiometry: Dict[str, float],
    biomass_rxn_id: str
) -> nx.DiGraph:
    """
    Add biomass objective function (BOF) features to metabolite nodes.
    
    Features added:
    - in_bof: 1 if metabolite is consumed by biomass reaction, 0 otherwise
    - bof_coef: stoichiometric coefficient in BOF (0 if not in BOF)
    
    Only metabolite nodes get these features (reactions get 0).
    
    Args:
        G: NetworkX DiGraph
        bof_stoichiometry: Dict of {metabolite_id: coefficient}
        biomass_rxn_id: ID of biomass reaction (for validation)
        
    Returns:
        Modified graph with BOF features
    """
    if biomass_rxn_id not in G:
        logger.warning(f"Biomass reaction {biomass_rxn_id} not in graph")
    
    bof_met_count = 0
    
    for node in G.nodes:
        node_type = G.nodes[node].get('node_type', -1)
        
        if node_type == 0:  # Metabolite
            if node in bof_stoichiometry:
                G.nodes[node]['in_bof'] = 1
                G.nodes[node]['bof_coef'] = float(bof_stoichiometry[node])
                bof_met_count += 1
            else:
                G.nodes[node]['in_bof'] = 0
                G.nodes[node]['bof_coef'] = 0.0
        else:  # Reaction
            G.nodes[node]['in_bof'] = 0
            G.nodes[node]['bof_coef'] = 0.0
    
    logger.info(f"Added BOF features: {bof_met_count} metabolites in biomass")
    
    return G


def compute_graph_features(
    G: nx.DiGraph,
    biomass_rxn_id: str,
    bof_stoichiometry: Dict[str, float]
) -> nx.DiGraph:
    """
    Compute and add all biomass-aware features to graph.
    
    This is the main entry point for feature computation.
    
    Features added:
    - dist_to_biomass (all nodes)
    - in_bof, bof_coef (metabolites only)
    
    Args:
        G: NetworkX DiGraph (from build_bipartite_graph)
        biomass_rxn_id: ID of biomass reaction
        bof_stoichiometry: Dict of {metabolite_id: coefficient}
        
    Returns:
        Graph with all features added
    """
    logger.info("Computing graph features...")
    
    # Add distance features
    G = add_distance_features(G, biomass_rxn_id)
    
    # Add BOF features
    G = add_bof_features(G, bof_stoichiometry, biomass_rxn_id)
    
    logger.info("Feature computation complete")
    
    return G