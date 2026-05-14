# Conformal Calibration Experiment — Design

**Date:** 2026-05-14
**Status:** SPEC (drives implementation of the experiment folder + core `src/growth` modules)
**Parent rationale:** `docs/CONFORMAL_PATH_ANALYSIS.md`

## 1. Question

The τ-sweep main experiment established a null: segmentation-derived σ²_v injected
into a heteroscedastic LME does not improve interval calibration over homoscedastic
LME at the empirical scale (`docs/CONFORMAL_PATH_ANALYSIS.md` §1). This experiment
asks a different question:

> Given the σ²_v null, can **distribution-free calibration** (conformal prediction)
> recover whatever adaptive *shape* information σ²_v carries, with a finite-sample
> coverage guarantee that the Gaussian-likelihood baseline lacks?

and a second, orthogonal question:

> Does a **proper Bayesian model average over the M=20 ensemble trajectories**
> (Path B) calibrate better than collapsing the ensemble to a scalar σ²_v offset?

The headline metric remains **IS@95** (Winkler interval score, `compute_interval_score`,
α=0.05). Coverage@95, mean CI width, R²_log, and CRPS are secondary.

## 2. The comparison grid

Three **base point-prediction models** (the y target is `uncertainty/logvol_mean`,
the ensemble-mean log-volume, for all three — the baseline is unchanged):

| Key            | Class                      | σ²_v source                                  |
|----------------|----------------------------|-----------------------------------------------|
| `lme_homo`     | `LMEGrowthModel`           | none (correct baseline)                       |
| `lme_hetero`   | `LMEHeteroGrowthModel`     | `uncertainty/men_mean_entropy` (mean-matched to the `logvol_var` scale), or `logvol_var` raw, injected as known measurement variance |
| `ensemble_bma` | `EnsembleLMEGrowthModel`   | M=20 per-member trajectories from `uncertainty/per_member_volumes` (Path B mixture) |

Four **calibration layers**, each applied on top of each base model:

| Key             | Method                                            | Spec ref |
|-----------------|---------------------------------------------------|----------|
| `parametric`    | the model's native Gaussian ±1.96σ interval       | status quo |
| `jackknife_plus`| Jackknife+ (Barber et al. 2021), signed score     | §3.2, §4.4 |
| `cqr_norm`      | Normalised conformity (Papadopoulos 2008): score `|y−μ̂|/σ̂`, reuses base σ̂ | §3.3a |
| `cqr_proper`    | CQR (Romano et al. 2019): two quantile regressors + conformalised max-score | §3.3b |

Grid = 3 base models × {parametric + 3 conformal} → 12 evaluated configurations.
One LOPO run of a base model produces **all four** calibration layers at once
(they share the nested-LOPO residual reservoir), so the manifest is **3 base
models × N seeds tasks**, not 12 × N.

`ensemble_bma` carries no per-scan σ̂ in the LMEHetero sense; for its `cqr_norm`
layer, σ̂ = the mixture predictive std (which already reflects member disagreement),
so `cqr_norm` on `ensemble_bma` is well-defined.

## 3. Evaluation: nested LOPO

Conformal coverage must be evaluated on patients not used for calibration.
`ConformalLOPOEvaluator` (`src/growth/evaluation/conformal_lopo.py`) does the
honest nested evaluation:

```
for held_out in patients (N folds):                      # outer LOPO
    train = the other N−1 patients
    # inner jackknife over the N−1 training patients:
    for i in range(N−1):
        μ̂_{−i} = base_model.fit(train \ {i})             # N−2 patients
        R_i  = signed LOO residual of patient i           # last_from_rest target
        σ̂_i = base_model predictive std at patient i      # for cqr_norm
    base = base_model.fit(train)                          # N−1 patients
    parametric  = base.predict(held_out)                  # native Gaussian interval
    jackknife+  = quantiles of {μ̂_{−i}(t*) ± R_i}         # Barber 2021 L/U
    cqr_norm    = base.predict(held_out) ± Q·σ̂(held_out)  # Q from {|R_i|/σ̂_i}
    cqr_proper  = CQR on an internal 2/3–1/3 split of train (seed-dependent)
```

Cost: `N² + N` base fits per (base model, seed). For LME at N≈54 this is ~2.5 min;
for `ensemble_bma` it is ×M ≈ 50 min — still trivial, and SLURM-parallelised one
task per (base model, seed).

**Granularity (doc §4.2):** patient-level exchangeability. The prediction target
is the `last_from_rest` protocol — last timepoint from all preceding — so each
patient contributes exactly **one** residual / one interval. This keeps the
exchangeability assumption clean (no within-patient temporal correlation leaks
into the calibration set).

## 4. Core functionality (lands in `src/growth`, formally)

| Module | Contents |
|--------|----------|
| `src/growth/shared/growth_models.py` | `PatientTrajectory.observation_ensemble: np.ndarray \| None` field, shape `[n_i, M]`, validated in `__post_init__`. |
| `src/growth/models/growth/ensemble_lme.py` | `EnsembleLMEGrowthModel(GrowthModel)` — Path B. Fits one `LMEGrowthModel` per ensemble member; `predict` returns the equal-weight Gaussian mixture: mean = mean_m μ_m, var = mean_m σ_m² + var_m μ_m (law of total variance); `lower_95`/`upper_95` from exact mixture-CDF quantile inversion; component (μ_m, σ_m²) in `metadata`. |
| `src/growth/shared/conformal.py` | `jackknife_plus_interval`, `NormalizedConformalCalibrator`, `SplitConformalCalibrator`, `CQRCalibrator`, `ConformalPredictiveSystemCalibrator`, `beta_binomial_coverage_ci`. crepes-backed where the math fits arrays directly; jackknife+ aggregation hand-rolled (Barber 2021 L/U formula — no library does the LOPO-coupled version). |
| `src/growth/evaluation/conformal_lopo.py` | `ConformalLOPOEvaluator` — the nested-LOPO orchestration above; returns `ConformalLOPOResults` with per-patient intervals for every calibration layer. |
| `src/growth/shared/bootstrap.py` | `cohen_d_paired`, `benjamini_hochberg`, `paired_bootstrap_ci` (BCa CI on the mean per-patient metric difference). |
| `src/growth/stages/stage1_volumetric/trajectory_loader.py` | `load_ensemble_trajectories_from_h5` — builds `PatientTrajectory` with `observation_ensemble` from `uncertainty/per_member_volumes` (transform: `log1p`; verified `mean_m log1p(V_m) == logvol_mean`). |

External library: **crepes ≥0.7** (`ConformalRegressor.fit(residuals, sigmas=)` +
`.predict_int(...)`; `ConformalPredictiveSystem` for proper predictive distributions).
Added to `pyproject.toml`.

## 5. Experiment folder layout (`experiments/stage1_volumetric/conformal_calibration/`)

Mirrors `main_experiment/` conventions:

```
conformal_calibration/
├── README.md, DESIGN.md, __init__.py
├── run.py                       # CLI: --write-manifest / --task-index K / --analyze / --smoke
├── configs/
│   ├── picasso.yaml             # production: signal = men_mean_entropy (mean-matched)
│   ├── picasso_variance.yaml    # ablation: signal = logvol_var (raw)
│   ├── local.yaml
│   └── local_smoke.yaml         # 2 seeds, n_patients=24, ensemble M=5
├── modules/
│   ├── cohort.py                # load_cohort: trajectories + observation_ensemble + σ²_v signal
│   ├── runner.py                # run_task: one (base_model, seed) → ConformalLOPOEvaluator → per-patient JSON
│   ├── statistics.py            # paired BCa bootstrap + Wilcoxon + Cohen's d + BH-FDR over the comparison family
│   ├── aggregator.py            # walk runs/, build long-form parquet/csv
│   └── figures.py               # IS@95 / coverage by (model × calibration); per-tertile panels; PIT grid
├── slurm/
│   ├── launcher.sh              # writes manifest, sbatch --array, then analysis_worker with afterany dep
│   ├── worker.sh                # one array task = one (base_model, seed)
│   └── analysis_worker.sh       # run.py --analyze
└── tests/                       # experiment-level: config schema, manifest size
```

Core algorithm tests live in `tests/growth/` (project convention + markers):
`test_ensemble_lme.py`, `test_conformal.py`, `test_conformal_lopo.py`.

## 6. Config schema (additions over `main_experiment`)

```yaml
experiment:   {name: conformal_calibration, seed: 42}
paths:        {mengrowth_h5, output_dir}
time:         {variable: ordinal, missing_date_strategy: skip}   # variable ∈ {ordinal, days_from_baseline}
uncertainty:  {estimator: mean_std, signal: men_mean_entropy, mean_signal: logvol_mean, scaling: mean_matched, floor_variance: 1e-6}
              # signal: men_mean_entropy (default) | logvol_var (ablation)
              # scaling: mean_matched (rescale a non-variance-unit signal to the logvol_var mean) | raw
volume:       {transform: identity}
patients:     {exclude, min_timepoints: 2, skip_all_zero_volume, max_logvol_std, max_patients: null}
ensemble:     {n_members: 20}                                    # M; smoke overrides to 5
models:       {lme_homo: true, lme_hetero: true, ensemble_bma: true}
conformal:
  alpha: 0.05
  layers: [parametric, jackknife_plus, cqr_norm, cqr_proper]
  jackknife_plus: {score: signed}                                # signed | symmetric
  cqr_proper:     {calib_fraction: 0.33, quantile_solver: highs}
evaluation:   {protocol: last_from_rest, n_seeds: 20, n_restarts: 5}
statistics:
  bootstrap:  {n_samples: 10000, confidence_level: 0.95, seed: 12345}
  wilcoxon: true
  cohens_d: true
  bh_fdr_q: 0.05
  comparison_families:
    calibration_lift:  # within each base model: each conformal layer vs parametric
    model_lift:        # within parametric: lme_hetero & ensemble_bma vs lme_homo
    headline:          # jackknife_plus@lme_homo vs parametric@lme_homo;
                       # cqr_norm@lme_hetero vs parametric@lme_hetero  (doc §7 stop conditions)
reporting:
  tertiles: sigma_v_sq        # stratify coverage & IS@95 by empirical σ²_v tertile (doc §4.3)
  figures: [is_by_model_calibration, coverage_by_model_calibration, tertile_panel, width_vs_sigmav]
slurm:        {partition, constraint: cpu, time, cpus_per_task, mem, conda_env, repo_dir, logs_dir, array_throttle}
```

Note `slurm.constraint: cpu` — every `sbatch` in `launcher.sh` MUST pass
`--constraint=cpu` explicitly (Picasso lua plugin overrides the worker's
`#SBATCH --constraint`).

## 7. Outputs

```
{output_dir}/runs/{base_model}/seed_{NNN}/
    conformal_lopo_results.json    # per-patient, per-layer intervals + actual + σ²_v_target
    marginal_metrics.json          # IS@95, cov95, R²_log, width, CRPS per calibration layer
    tertile_metrics.json           # the same, stratified by σ²_v tertile
{output_dir}/aggregated/
    results_table.parquet
    statistics.json                # bootstrap CIs, Wilcoxon p, Cohen's d, BH-FDR per family
{output_dir}/figures/*.png
{output_dir}/{manifest,cohort_meta}.json
```

## 8. day_from_baseline readiness

`time.variable` switches the loaders between `timepoint_idx` (ordinal, current)
and deltas derived from `metadata/study_date` (`days_from_baseline`). The H5
already carries `metadata/study_date`; `trajectory_loader._compute_deltas_from_dates`
handles it. No code path is ordinal-only — when real dates land, flipping the
config key reruns the whole experiment on continuous time. The smoke test and
production runs use `ordinal` until real dates are available.

## 9. Seeds

At τ=0 (empirical σ²_v, no sampling) the only seed-sensitive components are
(a) `LMEHeteroGrowthModel`'s multi-restart L-BFGS-B optimiser and (b) the
internal 2/3–1/3 split of `cqr_proper`. The seed axis therefore measures
optimiser stability + CQR split variance; for `lme_homo` + {parametric,
jackknife_plus} it is expected to be near-deterministic, which is itself a
reported sanity check. The statistics module treats seeds as replicates,
mirroring `main_experiment`.

## 10. Smoke test

`configs/local_smoke.yaml`: 2 seeds, `patients.max_patients: 12`,
`ensemble.n_members: 5`, all 3 base models × 4 layers, run sequentially via
`run.py --smoke`. Target wall-clock < 5 min. Verifies: every layer produces
finite intervals, the statistics suite emits bootstrap/Wilcoxon/Cohen's d/BH-FDR,
figures render.
