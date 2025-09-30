"""
Core graph construction from SBML metabolic models.
Builds directed bipartite graphs: metabolites <-> reactions.
"""
import networkx as nx
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def build_bipartite_graph(model, biomass_rxn_id: Optional[str] = None) -> nx.DiGraph:
    """
    Build a directed bipartite graph from a COBRApy model.
    
    Graph structure:
    - Nodes: Metabolites (type=0) + Reactions (type=1)
    - Edges: Metabolite -> Reaction (if reactant)
              Reaction -> Metabolite (if product)
    
    Node attributes (initialized):
    - Metabolites: node_type=0, met_id=<id>
    - Reactions: node_type=1, rxn_id=<id>, is_biomass=0/1, is_exchange=0/1
    
    Edge attributes:
    - stoich_coef: absolute stoichiometric coefficient
    - is_reactant: 1 if met->rxn edge, 0 if rxn->met edge
    
    Args:
        model: COBRApy model
        biomass_rxn_id: ID of biomass reaction (for is_biomass flag)
        
    Returns:
        NetworkX DiGraph with bipartite structure
    """
    G = nx.DiGraph()
    
    # Add metabolite nodes
    for met in model.metabolites:
        G.add_node(
            met.id,
            node_type=0,
            met_id=met.id,
            name=met.name if hasattr(met, 'name') else met.id,
            compartment=met.compartment if hasattr(met, 'compartment') else 'unknown'
        )
    
    logger.debug(f"Added {len(model.metabolites)} metabolite nodes")
    
    # Add reaction nodes
    for rxn in model.reactions:
        is_biomass = 1 if (biomass_rxn_id and rxn.id == biomass_rxn_id) else 0
        is_exchange = 1 if (getattr(rxn, 'boundary', False) or rxn.id.startswith('EX_')) else 0
        
        G.add_node(
            rxn.id,
            node_type=1,
            rxn_id=rxn.id,
            name=rxn.name if hasattr(rxn, 'name') else rxn.id,
            is_biomass=is_biomass,
            is_exchange=is_exchange,
            reversible=1 if rxn.reversibility else 0
        )
    
    logger.debug(f"Added {len(model.reactions)} reaction nodes")
    
    # Add edges based on stoichiometry
    edge_count = 0
    for rxn in model.reactions:
        for met, coef in rxn.metabolites.items():
            stoich_abs = abs(float(coef))
            
            if coef < 0:
                # Negative coefficient = reactant (consumed)
                # Edge: Metabolite -> Reaction
                G.add_edge(
                    met.id, 
                    rxn.id,
                    stoich_coef=stoich_abs,
                    is_reactant=1
                )
                edge_count += 1
            else:
                # Positive coefficient = product (produced)
                # Edge: Reaction -> Metabolite
                G.add_edge(
                    rxn.id,
                    met.id,
                    stoich_coef=stoich_abs,
                    is_reactant=0
                )
                edge_count += 1
    
    logger.debug(f"Added {edge_count} edges")
    
    # Validate graph structure
    if not nx.is_bipartite(G.to_undirected()):
        logger.warning("Graph is not bipartite (unexpected for metabolic networks)")
    
    logger.info(
        f"Built graph: {G.number_of_nodes()} nodes "
        f"({len(model.metabolites)} mets + {len(model.reactions)} rxns), "
        f"{G.number_of_edges()} edges"
    )
    
    return G


def validate_graph(G: nx.DiGraph) -> dict:
    """
    Validate graph structure and return statistics.
    
    Returns:
        Dict with validation results and statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_metabolites': sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 0),
        'num_reactions': sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 1),
        'num_biomass_rxns': sum(1 for _, d in G.nodes(data=True) if d.get('is_biomass') == 1),
        'num_exchange_rxns': sum(1 for _, d in G.nodes(data=True) if d.get('is_exchange') == 1),
        'is_connected': nx.is_weakly_connected(G),
        'num_components': nx.number_weakly_connected_components(G)
    }
    
    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    stats['num_isolated'] = len(isolated)
    
    # Check bipartite structure
    try:
        is_bip = nx.is_bipartite(G.to_undirected())
        stats['is_bipartite'] = is_bip
    except:
        stats['is_bipartite'] = False
    
    return stats