# Conservation-constraint load and language-model physics reasoning

This repository is the public reproducibility package for the manuscript
*Conservation-constraint load predicts language-model physics-reasoning failure*.

The project evaluates whether the number of independent scalar conservation constraints required by a physics problem, denoted `d_c`, predicts large language model failure after controlling for benchmark, topic, answer type, item length, model family and inference budget.

## What is included

- `code/scripts/`: deterministic analysis and figure-generation scripts.
- `evaluation/`: derived aggregate tables, model/fit summaries and audit reports used by the manuscript figures.
- `figures/`: generated display figures and cover-art candidate.
- `figure_inputs/`: minimal derived CSV inputs for the MLST formulation-mechanism
  and capability-gradient figures. These files do not redistribute raw
  third-party benchmark text.
- `derivation/`: supporting theoretical notes for the constraint-penalty framing.
- `DATASETS_AND_LINKS.csv`: source registry for third-party benchmarks and model references.
- `REPRODUCIBLE_RUNBOOK.md`: local rerun instructions.

## What is intentionally excluded

This repository does not redistribute raw third-party benchmark text, downloaded benchmark archives, active submission manuscripts, cover letters, private logs, API credentials or reviewer materials. Some model-response files can reproduce substantial benchmark text; those are excluded unless their source licences and terms clearly permit redistribution.

Use the original dataset sources listed in `DATASETS_AND_LINKS.csv` to regenerate raw inputs under the relevant terms.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Regenerate the main figures from the included derived tables:

```bash
python code/scripts/40_fig_constraint_penalty_law.py
python code/scripts/41_fig_construct_validity.py
python code/scripts/70_fig3_formulation_mechanism.py
python code/scripts/55_fig_mechanism_schematic.py
python code/scripts/61_fig_capability_gradient.py
```

More details are in `REPRODUCIBLE_RUNBOOK.md`.

## Citation

Please cite the associated manuscript and this repository. Metadata are provided in `CITATION.cff`.
