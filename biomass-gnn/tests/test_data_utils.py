"""
Unit tests for data utilities.
Run with: pytest tests/test_data_utils.py -v
"""
import pytest
import os
import tempfile
from src.data.sbml_utils import (
    read_sbml_model, 
    find_biomass_reaction,
    list_exchange_reactions,
    apply_media_bounds,
    get_bof_stoichiometry,
    get_model_info
)
from src.data.fba import run_fba
from src.data.media import load_media_dir, MediaProfile


@pytest.fixture
def test_model():
    """Create a test E. coli core model."""
    try:
        import cobra.test
        # Try new API first
        try:
            model = cobra.test.create_test_model("textbook")
        except AttributeError:
            # Fallback to loading from file if it exists
            test_model_path = "data/models/e_coli_core.xml"
            if os.path.exists(test_model_path):
                from src.data.sbml_utils import read_sbml_model
                model = read_sbml_model(test_model_path, validate=False)
            else:
                pytest.skip(f"No test model available. Run: python scripts/make_test_sbml.py")
        return model
    except Exception as e:
        pytest.skip(f"Could not create test model: {e}")


class TestSBMLUtils:
    
    def test_find_biomass_reaction(self, test_model):
        """Test biomass reaction detection."""
        biomass = find_biomass_reaction(test_model)
        assert biomass is not None, "Should find biomass reaction"
        assert isinstance(biomass, str), "Should return string ID"
        assert "biomass" in biomass.lower() or "bio" in biomass.lower()
    
    def test_list_exchange_reactions(self, test_model):
        """Test exchange reaction listing."""
        exchanges = list_exchange_reactions(test_model)
        assert len(exchanges) > 0, "Should find exchange reactions"
        assert all(isinstance(rxn_id, str) for rxn_id in exchanges)
        # Most exchange reactions start with EX_
        ex_prefix = sum(1 for r in exchanges if r.startswith("EX_"))
        assert ex_prefix > 0, "Should have EX_ prefixed reactions"
    
    def test_apply_media_bounds_preserve_defaults(self, test_model):
        """Test media application preserving defaults."""
        # Get original bounds
        ex_glc = test_model.reactions.get_by_id("EX_glc__D_e")
        original_lb = ex_glc.lower_bound
        
        # Apply media with preserve defaults
        media_bounds = {"EX_o2_e": -15.0}  # Only modify oxygen
        apply_media_bounds(test_model, media_bounds, default_lb=None)
        
        # Check oxygen changed
        ex_o2 = test_model.reactions.get_by_id("EX_o2_e")
        assert ex_o2.lower_bound == -15.0
        
        # Check glucose preserved
        assert ex_glc.lower_bound == original_lb
    
    def test_apply_media_bounds_close_all(self, test_model):
        """Test media application closing unlisted exchanges."""
        media_bounds = {"EX_glc__D_e": -10.0}
        apply_media_bounds(test_model, media_bounds, default_lb=0.0)
        
        # Glucose should be open
        ex_glc = test_model.reactions.get_by_id("EX_glc__D_e")
        assert ex_glc.lower_bound == -10.0
        
        # Other exchanges should be closed (or near 0)
        ex_o2 = test_model.reactions.get_by_id("EX_o2_e")
        assert ex_o2.lower_bound == 0.0
    
    def test_get_bof_stoichiometry(self, test_model):
        """Test BOF stoichiometry extraction."""
        biomass = find_biomass_reaction(test_model)
        stoich = get_bof_stoichiometry(test_model, biomass)
        
        assert isinstance(stoich, dict)
        assert len(stoich) > 0, "BOF should consume metabolites"
        assert all(isinstance(k, str) for k in stoich.keys())
        assert all(v > 0 for v in stoich.values()), "Coefficients should be positive"
    
    def test_get_model_info(self, test_model):
        """Test model info extraction."""
        info = get_model_info(test_model)
        
        assert 'num_reactions' in info
        assert 'num_metabolites' in info
        assert 'num_exchanges' in info
        assert 'biomass_rxn' in info
        
        assert info['num_reactions'] > 0
        assert info['num_metabolites'] > 0
        assert info['biomass_rxn'] is not None


class TestFBA:
    
    def test_run_fba_basic(self, test_model):
        """Test basic FBA execution."""
        flux = run_fba(test_model)
        assert isinstance(flux, float)
        assert flux >= 0, "Biomass flux should be non-negative"
        assert flux > 0.5, "E. coli core should grow under default conditions"
    
    def test_run_fba_with_objective(self, test_model):
        """Test FBA with explicit objective."""
        biomass = find_biomass_reaction(test_model)
        flux = run_fba(test_model, objective_rxn_id=biomass)
        assert flux > 0
    
    def test_run_fba_infeasible(self, test_model):
        """Test FBA with infeasible conditions."""
        # Close all exchanges
        for rxn in test_model.reactions:
            if rxn.id.startswith("EX_"):
                rxn.lower_bound = 0.0
                rxn.upper_bound = 0.0
        
        flux = run_fba(test_model)
        assert flux == 0.0, "Should return 0 for infeasible problem"
    
    def test_run_fba_timeout(self, test_model):
        """Test FBA with timeout (should still succeed for small model)."""
        flux = run_fba(test_model, timeout_seconds=10)
        assert flux >= 0


class TestMedia:
    
    def test_load_media_dir(self):
        """Test loading media profiles from directory."""
        # Create temporary media directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal media
            minimal_path = os.path.join(tmpdir, "minimal.csv")
            with open(minimal_path, 'w') as f:
                f.write("EX_glc__D_e,-10\n")
                f.write("EX_o2_e,-20\n")
            
            # Create rich media
            rich_path = os.path.join(tmpdir, "rich.csv")
            with open(rich_path, 'w') as f:
                f.write("EX_glc__D_e,-10\n")
                f.write("EX_o2_e,-20\n")
                f.write("EX_lys__L_e,-5\n")
            
            # Load media
            profiles = load_media_dir(tmpdir)
            
            assert len(profiles) == 2
            assert all(isinstance(p, MediaProfile) for p in profiles)
            
            # Check names
            names = [p.name for p in profiles]
            assert "minimal" in names
            assert "rich" in names
            
            # Check bounds
            minimal = next(p for p in profiles if p.name == "minimal")
            assert len(minimal.bounds) == 2
            assert "EX_glc__D_e" in minimal.bounds
            assert minimal.bounds["EX_glc__D_e"] == -10.0
            
            rich = next(p for p in profiles if p.name == "rich")
            assert len(rich.bounds) == 3


class TestIntegration:
    
    def test_full_label_generation_workflow(self, test_model):
        """Test complete workflow: model + media -> FBA -> label."""
        # Define media
        media_bounds = {
            "EX_glc__D_e": -10.0,
            "EX_o2_e": -20.0,
            "EX_nh4_e": -10.0,
            "EX_pi_e": -10.0
        }
        
        # Apply media (preserve defaults)
        apply_media_bounds(test_model, media_bounds, default_lb=None)
        
        # Find biomass
        biomass = find_biomass_reaction(test_model)
        assert biomass is not None
        
        # Run FBA
        flux = run_fba(test_model, objective_rxn_id=biomass)
        
        # Create label
        grow = 1 if flux > 1e-6 else 0
        
        assert flux > 0, "Should have positive growth"
        assert grow == 1, "Should predict growth"
        
        # Verify BOF stoichiometry
        stoich = get_bof_stoichiometry(test_model, biomass)
        assert len(stoich) > 0, "BOF should have metabolites"