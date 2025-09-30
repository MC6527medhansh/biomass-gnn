"""
Flux Balance Analysis (FBA) Runner
Enhanced with error handling and timeouts.
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def run_fba(
    model, 
    objective_rxn_id: Optional[str] = None,
    timeout_seconds: Optional[int] = 60
) -> float:
    """
    Run FBA and return the optimal objective value (biomass flux).
    
    Args:
        model: COBRApy model
        objective_rxn_id: Reaction ID to optimize (None = use model's current objective)
        timeout_seconds: Solver timeout in seconds (None = no timeout)
        
    Returns:
        Optimal objective value (typically biomass flux in 1/hr)
        Returns 0.0 if optimization fails or is infeasible
        
    Handles:
        - Infeasible solutions
        - Unbounded solutions  
        - Solver timeouts
        - Missing objectives
    """
    from .sbml_utils import _ensure_cobra
    _ = _ensure_cobra()
    
    # Set objective if specified
    if objective_rxn_id is not None:
        try:
            model.objective = objective_rxn_id
            logger.debug(f"Set objective to {objective_rxn_id}")
        except Exception as e:
            logger.warning(f"Could not set objective to {objective_rxn_id}: {e}")
            # Continue anyway - maybe model already has correct objective
    
    # Check that model has an objective
    try:
        obj_rxns = [v.name for v in model.objective.variables]
        if not obj_rxns:
            logger.error("Model has no objective reactions set")
            return 0.0
        logger.debug(f"Optimizing objective: {obj_rxns}")
    except Exception as e:
        logger.error(f"Could not determine objective: {e}")
        return 0.0
    
    # Configure solver timeout if specified
    if timeout_seconds:
        try:
            model.solver.configuration.timeout = timeout_seconds
        except Exception:
            logger.debug("Could not set solver timeout (solver may not support it)")
    
    # Run optimization
    try:
        solution = model.optimize()
        
        # Check solution status
        if solution.status == 'infeasible':
            logger.debug("FBA solution is infeasible (no feasible flux distribution)")
            return 0.0
        elif solution.status == 'unbounded':
            logger.warning("FBA solution is unbounded (model may have issues)")
            return 0.0
        elif solution.status != 'optimal':
            logger.warning(f"FBA solution status: {solution.status}")
            return 0.0
        
        # Extract objective value
        if solution.objective_value is not None:
            obj_val = float(solution.objective_value)
            logger.debug(f"FBA objective value: {obj_val:.6f}")
            return obj_val
        else:
            logger.warning("FBA returned no objective value")
            return 0.0
            
    except Exception as e:
        logger.error(f"FBA optimization failed: {e}")
        return 0.0


def run_fva(model, reaction_id: str, fraction: float = 0.9) -> tuple:
    """
    Run Flux Variability Analysis for a single reaction.
    
    Args:
        model: COBRApy model
        reaction_id: Reaction to analyze
        fraction: Fraction of optimal objective to constrain (0-1)
        
    Returns:
        Tuple of (min_flux, max_flux)
        Returns (0.0, 0.0) if analysis fails
    """
    from .sbml_utils import _ensure_cobra
    cobra = _ensure_cobra()
    
    try:
        fva_result = cobra.flux_analysis.flux_variability_analysis(
            model, 
            reaction_list=[reaction_id],
            fraction_of_optimum=fraction
        )
        return (
            float(fva_result.loc[reaction_id, 'minimum']),
            float(fva_result.loc[reaction_id, 'maximum'])
        )
    except Exception as e:
        logger.error(f"FVA failed for {reaction_id}: {e}")
        return (0.0, 0.0)