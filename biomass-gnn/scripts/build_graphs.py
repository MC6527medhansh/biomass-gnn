"""
Build bipartite metabolic network graphs from SBML models.

Usage:
    python scripts/build_graphs.py \\
        --models_dir data/models \\
        --out_dir data/graphs

Outputs:
    One .graphml file per model in out_dir/
"""
import argparse
import os
import glob
import logging
import networkx as nx
from src.data.sbml_utils import (
    read_sbml_model, 
    find_biomass_reaction, 
    get_bof_stoichiometry
)
from src.graph.builder import build_bipartite_graph, validate_graph
from src.graph.features import compute_graph_features

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description='Build metabolic network graphs from SBML models')
    ap.add_argument('--models_dir', required=True, help='Folder with SBML files')
    ap.add_argument('--out_dir', required=True, help='Output folder for GraphML files')
    args = ap.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all SBML files
    sbml_patterns = [
        os.path.join(args.models_dir, '*.xml'),
        os.path.join(args.models_dir, '*.sbml')
    ]
    sbmls = []
    for pattern in sbml_patterns:
        sbmls.extend(glob.glob(pattern))
    sbmls = sorted(set(sbmls))
    
    if not sbmls:
        logger.error(f"No SBML files found in {args.models_dir}")
        return
    
    logger.info(f"Found {len(sbmls)} SBML files")
    logger.info(f"Output directory: {args.out_dir}")
    
    # Process each model
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for model_path in sbmls:
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(args.out_dir, f"{model_id}.graphml")
        
        # Skip if already exists
        if os.path.exists(output_path):
            logger.info(f"[SKIP] {model_id} (already exists)")
            skip_count += 1
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model_id}")
        
        try:
            # Load model
            model = read_sbml_model(model_path, validate=True)
            
            # Find biomass reaction
            biomass_rxn = find_biomass_reaction(model)
            if biomass_rxn is None:
                logger.warning(f"No biomass reaction found in {model_id}; skipping")
                skip_count += 1
                continue
            
            logger.info(f"Biomass reaction: {biomass_rxn}")
            
            # Get BOF stoichiometry
            try:
                bof_stoich = get_bof_stoichiometry(model, biomass_rxn)
                logger.info(f"BOF contains {len(bof_stoich)} metabolites")
            except Exception as e:
                logger.warning(f"Could not extract BOF stoichiometry: {e}")
                bof_stoich = {}
            
            # Build graph
            G = build_bipartite_graph(model, biomass_rxn)
            
            # Add features
            G = compute_graph_features(G, biomass_rxn, bof_stoich)
            
            # Validate
            stats = validate_graph(G)
            logger.info(
                f"Graph: {stats['num_nodes']} nodes, "
                f"{stats['num_edges']} edges, "
                f"{stats['num_components']} components"
            )
            
            if stats['num_components'] > 1:
                logger.warning(f"Graph has {stats['num_components']} disconnected components")
            
            # Save to GraphML
            nx.write_graphml(G, output_path)
            logger.info(f"Saved: {output_path}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {model_id}: {e}")
            fail_count += 1
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY:")
    logger.info(f"  Successful: {success_count} graphs")
    logger.info(f"  Failed: {fail_count} models")
    logger.info(f"  Skipped: {skip_count} models")
    logger.info(f"  Output: {args.out_dir}")


if __name__ == '__main__':
    main()