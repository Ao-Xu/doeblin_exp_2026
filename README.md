# Doeblin Anchored Contrastive Learning Experiments

This directory contains the reproducible experiment pipeline for Section 8.
The main entry point is self-contained and does not import private modules.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the submission results

```bash
python run_experiments_v2.py --out results_v2 --seeds 10
```

The command regenerates all CSV files, LaTeX tables, and PDF figures used by
the paper in `results_v2/`.

For a faster smoke test, use:

```bash
python run_experiments_v2.py --out results_quick --seeds 3
```

The training-based figures are produced from end-to-end contrastive training,
de-anchoring, Markovization, and evaluation.  Deterministic diagnostics, such as
coverage failure, are labeled as diagnostics in the paper and in the generated
tables.
