# Reproducible Runbook

This runbook supports rerunning the public package without redistributing raw third-party benchmark text.

## Environment

Tested stack:

- Python 3.11.9
- NumPy 1.26.4
- pandas 2.2.2
- scikit-learn 1.4.2
- SciPy 1.13.1
- matplotlib 3.9.0

Install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, use `source .venv/bin/activate` instead of the Windows activation command.

## Included Derived Analyses

The `evaluation/` directory contains derived tables and reports that do not include raw benchmark problem statements. These are sufficient to inspect and regenerate the paper figures:

- `W6_controlled_dc_analysis.md`
- `w6_controlled_panel.csv`
- `w6_logit_coefficients.csv`
- `w6_controlled_marginal_by_dc.csv`
- `regex_dc_w6_robustness_20260529.md`
- `physreason_theorem_anchor_20260529.md`
- `rule_based_dc_floor_20260529.md`
- `C4_*`, `C5_*`, `reasoner_*` compute-intervention summaries

The `figure_inputs/` directory contains the minimal derived CSV inputs needed
for the MLST formulation-mechanism and capability-gradient figures. Sanitised
capability-gradient inputs retain only item IDs, `d_c` and correctness columns
needed by the plotting script.

## Regenerate Figures

```bash
python code/scripts/40_fig_constraint_penalty_law.py
python code/scripts/41_fig_construct_validity.py
python code/scripts/70_fig3_formulation_mechanism.py
python code/scripts/55_fig_mechanism_schematic.py
python code/scripts/61_fig_capability_gradient.py
```

Expected outputs are written to `figures/`.

## Rerun Core Analyses From Derived Tables

Some scripts consume derived panels already included in `evaluation/`:

```bash
python code/scripts/17_shape_competition.py --help
python code/scripts/20_w6_holdout_cv.py --help
python code/scripts/21_hierarchical_envelope_fit.py --help
python code/scripts/34_logodds_linearity_test.py --help
```

Scripts that require raw third-party benchmark files or model-response logs are included for transparency but will need local source data reconstructed from the datasets listed in `DATASETS_AND_LINKS.csv`.

## Reconstruct Raw Inputs

1. Read `DATASETS_AND_LINKS.csv`.
2. Download the relevant source datasets from their original repositories under their licences/terms.
3. Recreate the local project layout expected by the scripts, typically `data/benchmarks/...` and `data/results/...`.
4. Do not commit raw third-party benchmark text, downloaded archives, credentials or active manuscript files to this public repository.

## Reproducibility Boundary

This package supports verification of the reported analyses through released code, derived tables, figure inputs and source registries. Exact reruns of provider-hosted LLM inference may vary because hosted model aliases, snapshots, agent interfaces and safety filters can change; the observed API/model snapshot notes and as-run frontier interface labels are recorded in `evaluation/model_snapshot_metadata_2026-05.md`.
