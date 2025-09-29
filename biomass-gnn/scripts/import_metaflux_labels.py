"""
Convert a MetaFlux results CSV into our standard labels.csv format.

Expected input CSV (minimal):
    media_id,biomass_flux
    minimal,0.2134
    rich,0.9842
    anaer,0.0000

If your file has different column names, you can pass --media_col and --flux_col.

Usage:
  python scripts/import_metaflux_labels.py \
      --model_id ecoli_K12 \
      --metaflux_csv path/to/metaflux_results.csv \
      --out data/labels.csv \
      --append
"""
import argparse, csv, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="Name to use for the model_id column (e.g., ecoli_K12)")
    ap.add_argument("--metaflux_csv", required=True, help="CSV exported from MetaFlux (media vs biomass)")
    ap.add_argument("--out", required=True, help="Output CSV (usually data/labels.csv)")
    ap.add_argument("--append", action="store_true", help="Append to existing labels.csv if present")
    ap.add_argument("--media_col", default="media_id", help="Column name in metaflux_csv for media")
    ap.add_argument("--flux_col", default="biomass_flux", help="Column name in metaflux_csv for biomass value")
    args = ap.parse_args()

    rows = []
    with open(args.metaflux_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if args.media_col not in reader.fieldnames or args.flux_col not in reader.fieldnames:
            raise ValueError(f"Input CSV must have columns '{args.media_col}' and '{args.flux_col}'. Found: {reader.fieldnames}")
        for r in reader:
            media = r[args.media_col].strip()
            try:
                flux = float(r[args.flux_col])
            except Exception:
                continue
            grow = 1 if flux > 0 else 0
            rows.append([args.model_id, media, flux, grow])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = True
    if args.append and os.path.exists(args.out):
        # If file exists and has header, don't rewrite header
        with open(args.out, "r", newline="") as f:
            first = f.readline()
            if first and "model_id,media_id,biomass_flux,grow" in first.replace(" ", ""):
                write_header = False

    mode = "a" if args.append and os.path.exists(args.out) else "w"
    with open(args.out, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model_id","media_id","biomass_flux","grow"])
        for model_id, media, flux, grow in rows:
            w.writerow([model_id, media, flux, grow])
    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()