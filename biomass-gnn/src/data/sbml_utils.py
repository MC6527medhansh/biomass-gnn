from typing import Optional, Dict, List

def _ensure_cobra():
    try:
        import cobra  # noqa: F401
        return cobra
    except Exception as e:
        raise RuntimeError(
            "COBRApy is required. Install it first: pip install cobra\n"
            f"Original import error: {e}"
        )

def read_sbml_model(path: str):
    """Load an SBML model from disk using COBRApy."""
    cobra = _ensure_cobra()
    from cobra.io import read_sbml_model
    model = read_sbml_model(path)
    return model

def find_biomass_reaction(model) -> Optional[str]:
    """Try to find the biomass reaction ID."""
    # 1) Try objective variable, if set
    try:
        # model.objective._expression holds LinearExpression; below is robust way:
        # newer cobra: model.objective.expression.as_symbols()
        rxns = [v.name for v in model.objective.variables]
        if rxns:
            return rxns[0]
    except Exception:
        pass

    # 2) Fallback by name pattern
    for rxn in model.reactions:
        rid = rxn.id.lower()
        if "biomass" in rid or rid.startswith("bio"):
            return rxn.id
    return None

def list_exchange_reactions(model) -> List[str]:
    """Return IDs of exchange/boundary reactions."""
    ex = [rxn.id for rxn in model.reactions if getattr(rxn, 'boundary', False)]
    if not ex:
        ex = [rxn.id for rxn in model.reactions if rxn.id.startswith("EX_")]
    return ex

def apply_media_bounds(model, media_bounds: Dict[str, float], default_lb: float = 0.0):
    """
    Apply media lower bounds to exchange reactions.
    Convention: negative lower bound = allowed import (uptake).
    Everything else defaults to lb=0 (no import) unless specified.
    """
    for rxn in model.reactions:
        if getattr(rxn, 'boundary', False) or rxn.id.startswith("EX_"):
            lb = media_bounds.get(rxn.id, default_lb)
            try:
                rxn.lower_bound = lb
            except Exception:
                # Some models may lock bounds; ignore here.
                pass

def get_bof_stoichiometry(model, biomass_rxn_id: str) -> Dict[str, float]:
    """
    Return {metabolite_id: coefficient} for BOF reactants (consumed by biomass).
    Coefficients returned as positive numbers.
    """
    rxn = model.reactions.get_by_id(biomass_rxn_id)
    sto = {}
    for met, coef in rxn.metabolites.items():
        if coef < 0:  # negative -> reactant
            sto[met.id] = abs(float(coef))
    return sto