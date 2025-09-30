"""
Generate labels (biomass flux values) via FBA for all (model, media) combinations.

Usage:
    python scripts/generate_labels.py \\
        --models_dir data/models \\
        --media_dir data/media \\
        --out data/labels.csv

Output CSV format:
    model_id,media_id,biomass_flux,grow
"""
import argparse
import os
import glob
import csv
import logging
from src.data.media import load_media_dir
from src.data.sbml_utils import read_sbml_model, find_biomass_reaction, apply_media_bounds
from src.data.fba import run_fba

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description='Generate biomass flux labels via FBA')
    ap.add_argument('--models_dir', required=True, help='Folder with SBML files (*.xml or *.sbml)')
    ap.add_argument('--media_dir', required=True, help='Folder with media CSVs (e.g., minimal.csv)')
    ap.add_argument('--out', required=True, help='Output CSV path, e.g., data/labels.csv')
    ap.add_argument('--preserve_defaults', action='store_true', default=True,
                    help='Preserve model default bounds for unlisted exchanges (recommended)')
    ap.add_argument('--timeout', type=int, default=60, help='FBA solver timeout in seconds')
    args = ap.parse_args()

    # Load media profiles
    logger.info(f"Loading media profiles from {args.media_dir}")
    media_profiles = load_media_dir(args.media_dir)
    if not media_profiles:
        logger.error(f"No media CSV files found in {args.media_dir}")
        return
    logger.info(f"Loaded {len(media_profiles)} media profiles: {[m.name for m in media_profiles]}")

    # Create output directory
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # Find all SBML files
    sbml_patterns = [
        os.path.join(args.models_dir, '*.xml'),
        os.path.join(args.models_dir, '*.sbml')
    ]
    sbmls = []
    for pattern in sbml_patterns:
        sbmls.extend(glob.glob(pattern))
    sbmls = sorted(set(sbmls))  # Remove duplicates
    
    if not sbmls:
        logger.error(f"No SBML files found in {args.models_dir}")
        logger.info("Run: python scripts/make_test_sbml.py")
        return
    
    logger.info(f"Found {len(sbmls)} SBML files")

    # Open output CSV
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'media_id', 'biomass_flux', 'grow'])
        
        # Process each model
        for model_path in sbmls:
            model_id = os.path.splitext(os.path.basename(model_path))[0]
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {model_id}")
            
            # Load model
            try:
                model = read_sbml_model(model_path, validate=True)
            except Exception as e:
                logger.error(f"Failed to load {model_id}: {e}")
                fail_count += 1
                continue

            # Find biomass reaction
            biomass_rxn = find_biomass_reaction(model)
            if biomass_rxn is None:
                logger.warning(f"No biomass reaction found in {model_id}; skipping")
                skip_count += 1
                continue
            logger.info(f"Biomass reaction: {biomass_rxn}")

            # Test each media condition
            for media in media_profiles:
                logger.info(f"  Testing media: {media.name}")
                
                # Reload fresh model (to reset bounds)
                try:
                    m = read_sbml_model(model_path, validate=False)
                except Exception as e:
                    logger.error(f"    Failed to reload model: {e}")
                    continue
                
                # Apply media bounds
                # CRITICAL: Use default_lb=None to preserve model's original bounds
                apply_media_bounds(m, media.bounds, default_lb=None)
                
                # Run FBA
                flux = run_fba(m, objective_rxn_id=biomass_rxn, timeout_seconds=args.timeout)
                grow = 1 if flux > 1e-6 else 0  # Use small threshold to avoid floating point issues
                
                # Write result
                writer.writerow([model_id, media.name, f"{flux:.6f}", grow])
                
                # Log result
                status = "✓ GROW" if grow else "✗ NO GROWTH"
                logger.info(f"    {status} | biomass_flux={flux:.6f}")
                
                success_count += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY:")
    logger.info(f"  Successful: {success_count} labels")
    logger.info(f"  Failed: {fail_count} models")
    logger.info(f"  Skipped: {skip_count} models (no biomass reaction)")
    logger.info(f"  Output: {args.out}")
    
    # Check for problematic labels
    if success_count > 0:
        with open(args.out, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            grow_count = sum(1 for r in rows if r['grow'] == '1')
            grow_pct = 100 * grow_count / len(rows)
            logger.info(f"  Growth rate: {grow_pct:.1f}% ({grow_count}/{len(rows)} samples)")
            
            if grow_pct < 30:
                logger.warning(f"  ⚠️  Low growth rate detected!")
                logger.warning(f"     Consider checking media definitions or using --preserve_defaults")


if __name__ == '__main__':
    main()