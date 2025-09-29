from typing import Optional
from .sbml_utils import _ensure_cobra

def run_fba(model, objective_rxn_id: Optional[str] = None) -> float:
    """Run FBA and return the optimal objective (biomass flux)."""
    _ = _ensure_cobra()
    if objective_rxn_id is not None:
        try:
            model.objective = objective_rxn_id
        except Exception:
            # Not fatal; many models accept setting by string id
            pass
    solution = model.optimize()
    return float(solution.objective_value) if solution and solution.objective_value is not None else 0.0
