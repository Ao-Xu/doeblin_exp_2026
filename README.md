# Doeblin Anchored Contrastive Learning Experiments

This directory contains the reproducible experiment pipeline for Section 8.
The main entry point is self-contained and does not import private modules.

## Setup from the repository root

```bash
python -m pip install -r experiments/requirements.txt
```

## Run the submission results

```bash
cd experiments
python run_experiments_v2.py --out results_v2 --seeds 10
```

The command regenerates all CSV files, LaTeX tables, and PDF figures used by
the paper in `results_v2/`.

For a faster smoke test, use:

```bash
cd experiments
python run_experiments_v2.py --out results_quick --seeds 3
```

If your shell is already inside `experiments/`, omit the `cd experiments` line.

The training-based figures are produced from end-to-end contrastive training,
de-anchoring, Markovization, and evaluation.  Deterministic diagnostics, such as
coverage failure, are labeled as diagnostics in the paper and in the generated
tables.

## Theory Estimator / Implemented Estimator / Evaluated Quantity

The public script implements the estimator described in the paper.

| Component | Paper estimator | Public implementation |
|---|---|---|
| Positive samples | Weighted double-positive risk `(1-eps) ell+(X,Y) + eps ell+(X,Y_tilde)` | Same weighted double-positive construction; no Bernoulli mixture thinning is used. |
| Score class | Bounded candidate functions `gamma <= a <= Gamma` with posterior `a/(a + tau r)` | Bounded anchored-score network `a_theta = gamma + (Gamma-gamma) sigmoid(f_theta)`; the logistic logit is `log(a_theta/(tau r))`. |
| Validation excess | Held-out contrastive excess risk relative to the known synthetic oracle `a0` | Independent weighted validation blocks compute `R_val(a_hat) - R_val(a0)`; grid density errors are oracle diagnostics. |
| Reference and Markovization | Restart density with lower envelope and valid-kernel repair | Sampler and density evaluator are matched for each reference law; the poor-coverage reference is `0.1 Uniform + 0.9 Beta`; continuous-state integrals use grid quadrature. |

The score bounds used in the paper runs are `gamma=1e-4` and `Gamma=80`.

## Fixed Implementation Settings

| Item | Submission setting |
|---|---|
| Score network | ReLU MLP on `(x,y)` with bounded output `a_theta = gamma + (Gamma-gamma) sigmoid(f_theta)`, `gamma=1e-4`, `Gamma=80`. The default architecture is depth `2` and width `32`; the under-capacity ablation uses width `16`, and the runtime diagnostic uses width `24`. |
| Optimization | Adam with learning rate `2e-3`, batch size `512`, and weight decay `1e-5`. Unless explicitly varied, epochs are `10` for `n <= 1500`, `8` for `n <= 4000`, and `6` for larger `n`; selected stress tests use `7` epochs and runtime diagnostics use `5` epochs. |
| Contrastive construction | Weighted double-positive risk with default `eps=0.1` and `tau=5`; negative-sample ablations use `tau in {1,5,15}`. |
| Reference laws | Uniform restart law; poor-coverage stress test `0.1 Uniform + 0.9 Beta(2,8)`; empirical wrapped KDE with bandwidth `0.055`; and `0.5 Uniform + 0.5 KDE`. Samplers and density evaluators are matched. |
| Quadrature | Continuous-state Markovization and TV diagnostics use common grids, typically `54` design points and `54` to `72` next-state points; runtime diagnostics use `42 x 42` grids. |
| Trajectory diagnostics | Lazy finite-state chains with `alpha in {0.02,0.05,0.1,0.2,0.5}` and thinning `q in {1,2,5,10,20,50,100}`. |
| Randomness and timing | Main Monte Carlo summaries use ten seeds. Runtime diagnostics record measured wall-clock components on the local CPU submission environment used for the paper runs; they are not systems benchmarks. |

## Experiment Index

| Paper item | Interface checked | Main output files |
|---|---|---|
| Figure 1: statistical diagnostics | Calibration and one-dimensional rate decay | `fig1_end_to_end_calibration.pdf`, `exp1_end_to_end.csv`, `exp3_rates_dimension.csv` |
| Figure 2: Markovization | De-anchored scores can be invalid; Markovization restores kernel validity | `fig2_markovization_learned.pdf`, `exp2_markovization.csv` |
| Figure 3: anchor/reference | Anchor-strength and reference-coverage tradeoffs | `fig4_anchor_reference.pdf`, `exp4_anchor_reference.csv` |
| Figure 4: trajectory stress test | Temporal dependence and thinning diagnostics | `fig5_trajectory_real.pdf`, `exp5_trajectory_real.csv`, `exp5_thinning_real.csv`, `table6_thinning_effective_sample.tex` |
| Figure 5 and coverage table | Finite-horizon transfer and coverage failure | `fig6_dynamic_transfer_learned.pdf`, `exp6_dynamic_learned.csv`, `exp6_rare.csv`, `table7_coverage_failure.tex` |
| Figure 6 and ablation tables | Component necessity, numerical validity, and theory coverage | `fig7_ablation_heatmap.pdf`, `table3_method_comparison.tex`, `table4_ablation.tex`, `exp7_ablation.csv` |
| Figure 7: runtime | Measured cost of data construction, training, Markovization, and evaluation | `fig8_runtime_scalability.pdf`, `exp8_runtime.csv` |

The generated files `fig3_rates_slopes.pdf`, `table1_theory_map.tex`, and
`table2_models.tex` are retained as machine-generated provenance for the rate
diagnostic, experiment index, and model grid, but they are not included directly in the
paper body.

## Synthetic Models and Parameter Grids

| Module | Synthetic models | Main parameters |
|---|---|---|
| Calibration | Smooth, multimodal, and rough wrapped transition kernels | `n=800,1600,3200,6400`; 10 seeds |
| Markovization | Trained anchored contrastive scores on smooth, rough, and multimodal kernels | `n=1000` to `6400`; pre/post Markovization diagnostics |
| Rates | Real-trained one-dimensional smoothness models | `n=800` to `10000`; log-log slopes from trained runs only |
| Anchor/reference | Uniform, poor-coverage mixture, empirical KDE, and KDE-uniform mixture references | `eps=0.01,0.03,0.05,0.1,0.2,0.4,0.7` |
| Trajectory | Lazy finite-state chains and i.i.d. transition-pair references | mixing `alpha=0.02,0.05,0.1,0.2,0.5`; thinning `q=1,2,5,10,20,50,100` |
| Dynamics | Learned contrastive kernels and rare-state coverage failures | horizons `1,2,5,10,20,50`; rare-state mass `delta=0.01,0.02,0.05` |
| Ablation | No-anchor, no-deanchor, no-Markov, poor-reference, anchor-strength, capacity, and negative-sample variants | Same evaluation grids as the method comparison |
| Runtime | Continuous contrastive training and finite-state Markovization | `n=800,1600,3200,6400`, negatives `M=3,5,10`, finite states `S=50,100,200,500` |

## Implementation Notes

- Continuous-state TV and Markovization integrals are computed by grid quadrature on
  common evaluation grids.
- In the ablation heatmap, `Pre-M NegMass` is plotted as an absolute value on a
  log scale because the full method has zero negative mass to numerical precision.
  Other heatmap columns are relative to the full method.
- Path-law TV is reported as an occupancy-weighted perturbation upper bound, not as an
  exact path-law enumeration in continuous-state experiments.
- The trajectory experiment isolates temporal dependence; it is not a sharp
  dependent-data rate claim and does not analyze full-trajectory ERM.
- `--seeds 10` is the paper setting.  `--seeds 3` is a smoke-test mode for quicker
  reproduction.
