"""
SBML Model Utilities - Fixed Version
Handles SBML loading, biomass detection, and media application.
"""
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def _ensure_cobra():
    """Ensure COBRApy is installed."""
    try:
        import cobra
        return cobra
    except ImportError as e:
        raise RuntimeError(
            "COBRApy is required. Install it: pip install cobra\n"
            f"Original error: {e}"
        )


def read_sbml_model(path: str, validate: bool = True):
    """
    Load an SBML model from disk using COBRApy.
    
    Args:
        path: Path to SBML file
        validate: If True, perform basic validation checks
        
    Returns:
        cobra.Model instance
        
    Raises:
        FileNotFoundError: If path doesn't exist
        RuntimeError: If model loading fails
    """
    cobra = _ensure_cobra()
    from cobra.io import read_sbml_model
    
    try:
        model = read_sbml_model(path)
        
        if validate:
            if len(model.reactions) == 0:
                raise RuntimeError(f"Model {path} has no reactions")
            if len(model.metabolites) == 0:
                raise RuntimeError(f"Model {path} has no metabolites")
                
        logger.info(f"Loaded {path}: {len(model.reactions)} rxns, {len(model.metabolites)} mets")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load SBML from {path}: {e}")


def find_biomass_reaction(model) -> Optional[str]:
    """
    Find the biomass reaction ID using multiple strategies.
    
    Strategy:
    1. Check model.objective (most reliable)
    2. Search for reactions with 'biomass' in ID
    3. Search for reactions starting with 'bio'
    
    Returns:
        Reaction ID string, or None if not found
    """
    # Strategy 1: Check objective
    try:
        # Get reaction objects from objective, not names
        obj_rxns = list(model.objective.variables)
        if obj_rxns:
            # Extract the actual reaction ID
            rxn_id = obj_rxns[0].name
            # Verify this reaction actually exists in model
            if rxn_id in [r.id for r in model.reactions]:
                logger.debug(f"Found biomass from objective: {rxn_id}")
                return rxn_id
            else:
                logger.debug(f"Objective rxn {rxn_id} not in model, trying pattern search")
    except Exception as e:
        logger.debug(f"Could not extract from objective: {e}")
    
    # Strategy 2: Search by name pattern
    candidates = []
    for rxn in model.reactions:
        rid_lower = rxn.id.lower()
        if "biomass" in rid_lower:
            candidates.append((rxn.id, 2))  # priority 2
        elif rid_lower.startswith("bio"):
            candidates.append((rxn.id, 1))  # priority 1
    
    if candidates:
        # Sort by priority (higher first), then alphabetically
        candidates.sort(key=lambda x: (-x[1], x[0]))
        logger.debug(f"Found biomass candidates: {[c[0] for c in candidates]}")
        return candidates[0][0]
    
    logger.warning(f"No biomass reaction found in model with {len(model.reactions)} reactions")
    return None


def list_exchange_reactions(model) -> List[str]:
    """
    Return IDs of exchange/boundary reactions.
    
    Returns:
        List of reaction IDs
    """
    # Method 1: Use boundary attribute
    ex = [rxn.id for rxn in model.reactions if getattr(rxn, 'boundary', False)]
    
    # Method 2: Fallback to EX_ prefix
    if not ex:
        ex = [rxn.id for rxn in model.reactions if rxn.id.startswith("EX_")]
    
    logger.debug(f"Found {len(ex)} exchange reactions")
    return ex


def apply_media_bounds(
    model, 
    media_bounds: Dict[str, float], 
    default_lb: Optional[float] = None,
    strict: bool = False
):
    """
    Apply media lower bounds to exchange reactions.
    
    CRITICAL DESIGN DECISION:
    - If default_lb is None (RECOMMENDED): preserve model's original bounds
      Only modify exchanges explicitly listed in media_bounds.
      This allows models to work with their default media assumptions.
      
    - If default_lb is set (e.g., 0.0): close all unlisted exchanges.
      This gives full control but requires comprehensive media definitions.
    
    Args:
        model: COBRApy model
        media_bounds: Dict mapping exchange_rxn_id -> lower_bound
                     Negative values = uptake allowed
        default_lb: Default bound for unlisted exchanges (None = preserve original)
        strict: If True, raise error if media_bounds contains unknown reaction IDs
        
    Convention:
        Negative lower bound = uptake allowed (e.g., -10 = up to 10 mmol/gDW/hr uptake)
        Zero lower bound = no uptake (closed)
    """
    applied_count = 0
    missing_rxns = []
    
    # Get all exchange reactions
    exchange_ids = set(list_exchange_reactions(model))
    
    # Check for unknown reactions in media_bounds
    unknown = set(media_bounds.keys()) - exchange_ids
    if unknown:
        if strict:
            raise ValueError(f"Media contains unknown reactions: {unknown}")
        else:
            logger.warning(f"Media contains {len(unknown)} unknown reactions (will be ignored)")
            missing_rxns = list(unknown)
    
    # Apply bounds
    for rxn in model.reactions:
        if rxn.id not in exchange_ids:
            continue
            
        if rxn.id in media_bounds:
            # Apply specified bound
            try:
                old_lb = rxn.lower_bound
                rxn.lower_bound = media_bounds[rxn.id]
                applied_count += 1
                logger.debug(f"  {rxn.id}: {old_lb} -> {media_bounds[rxn.id]}")
            except Exception as e:
                logger.warning(f"Could not set bound for {rxn.id}: {e}")
                
        elif default_lb is not None:
            # Apply default bound (closes everything not listed)
            try:
                rxn.lower_bound = default_lb
            except Exception as e:
                logger.debug(f"Could not set default bound for {rxn.id}: {e}")
        # else: keep original bound
    
    logger.info(f"Applied {applied_count}/{len(media_bounds)} media bounds to model")
    if missing_rxns:
        logger.debug(f"Missing reactions: {missing_rxns[:5]}..." if len(missing_rxns) > 5 else f"Missing: {missing_rxns}")


def get_bof_stoichiometry(model, biomass_rxn_id: str) -> Dict[str, float]:
    """
    Extract biomass objective function (BOF) stoichiometry.
    
    Returns dict of {metabolite_id: coefficient} for reactants (consumed).
    Coefficients are returned as positive numbers.
    
    Args:
        model: COBRApy model
        biomass_rxn_id: ID of biomass reaction
        
    Returns:
        Dict mapping metabolite IDs to stoichiometric coefficients (positive)
        
    Raises:
        KeyError: If biomass_rxn_id not found in model
    """
    try:
        rxn = model.reactions.get_by_id(biomass_rxn_id)
    except KeyError:
        raise KeyError(f"Biomass reaction '{biomass_rxn_id}' not found in model")
    
    stoich = {}
    for met, coef in rxn.metabolites.items():
        if coef < 0:  # Negative = reactant (consumed)
            stoich[met.id] = abs(float(coef))
    
    logger.debug(f"Extracted {len(stoich)} metabolites from BOF")
    return stoich


def get_model_info(model) -> Dict:
    """
    Extract summary information about a model.
    
    Returns:
        Dict with keys: num_reactions, num_metabolites, num_genes, 
                       num_exchanges, biomass_rxn
    """
    biomass = find_biomass_reaction(model)
    exchanges = list_exchange_reactions(model)
    
    return {
        'num_reactions': len(model.reactions),
        'num_metabolites': len(model.metabolites),
        'num_genes': len(model.genes),
        'num_exchanges': len(exchanges),
        'biomass_rxn': biomass,
        'model_id': model.id if hasattr(model, 'id') else 'unknown'
    }