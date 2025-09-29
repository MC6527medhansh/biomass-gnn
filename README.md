# Biomass Prediction with Graph Neural Networks

Goal: predict **biomass (growth)** from a metabolic SBML model + nutrient media.

## Pipeline (high-level)
1) Generate labels with FBA (biomass flux, and growth yes/no).
2) Convert SBML to a bipartite graph (metabolites ↔ reactions).
3) Add simple features (distance-to-biomass, in-BOF flags, degrees, media vector).
4) Train baselines (tabular) and a graph-level GNN (regression and/or binary).
5) Evaluate on held-out organisms and media; run sanity checks.

This repo will be built step-by-step. Start in `configs/`, `data/`, and `src/`.
