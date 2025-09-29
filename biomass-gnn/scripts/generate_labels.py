import argparse, os, glob, csv
from src.data.media import load_media_dir
from src.data.sbml_utils import read_sbml_model, find_biomass_reaction, apply_media_bounds
from src.data.fba import run_fba

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', required=True, help='Folder with SBML files (*.xml or *.sbml)')
    ap.add_argument('--media_dir', required=True, help='Folder with media CSVs (e.g., minimal.csv)')
    ap.add_argument('--out', required=True, help='Output CSV path, e.g., data/labels.csv')
    args = ap.parse_args()

    media_profiles = load_media_dir(args.media_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        # header
        w.writerow(['model_id','media_id','biomass_flux','grow'])
        # loop over all SBML models
        sbmls = sorted(glob.glob(os.path.join(args.models_dir, '*.xml')) +
                       glob.glob(os.path.join(args.models_dir, '*.sbml')))
        if not sbmls:
            print(f"[WARN] No SBML files found in {args.models_dir}. Put some models there and retry.")
        for model_path in sbmls:
            model_id = os.path.splitext(os.path.basename(model_path))[0]
            try:
                model = read_sbml_model(model_path)
            except Exception as e:
                print(f"[ERROR] Could not load {model_id}: {e}")
                continue

            biomass_rxn = find_biomass_reaction(model)
            if biomass_rxn is None:
                print(f"[WARN] No biomass reaction found in {model_id}; skipping.")
                continue

            # run each media profile
            for media in media_profiles:
                # reload a fresh copy (so bounds reset each time)
                m = read_sbml_model(model_path)
                apply_media_bounds(m, media.bounds, default_lb=0.0)
                flux = run_fba(m, objective_rxn_id=biomass_rxn)
                grow = 1 if flux > 0 else 0
                w.writerow([model_id, media.name, flux, grow])
                print(f"{model_id} | {media.name} -> biomass={flux:.6f} grow={grow}")

    print(f"Saved labels to {args.out}")

if __name__ == '__main__':
    main()
