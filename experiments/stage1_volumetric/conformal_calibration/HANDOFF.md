# Conformal Calibration — Implementation Handoff

**Date:** 2026-05-14
**Status:** Implemented, smoke-tested locally. Not yet run at full scale (N≈54) or on Picasso.
**Parent docs:** `DESIGN.md` (spec), `README.md` (usage), `../../../docs/CONFORMAL_PATH_ANALYSIS.md` (scientific rationale).

This document is a complete pick-up point: what was built, where, why, how it is wired,
what is verified, and what remains.

---

## 1. TL;DR — what this experiment is

The τ-sweep `main_experiment` established a null: segmentation-derived σ²_v injected into a
heteroscedastic LME does not improve interval calibration over homoscedastic LME at the
empirical scale. This experiment is the salvage path. It compares, under LOPO-CV with the
**IS@95** headline metric:

- **`lme_homo`** — homoscedastic LME on the ensemble-mean log-volume trajectory (the *correct
  baseline*, unchanged).
- **`lme_hetero`** — LME with a segmentation-derived σ²_v injected as known measurement
  variance (Ensemble-derived heteroscedastic LME).
- **`ensemble_bma`** — Path B: one LME fitted per ensemble member, combined into an
  equal-weight Gaussian mixture (ensemble-of-trajectories Bayesian model averaging).

Each base model is then wrapped in **four calibration layers**: `parametric` (native
Gaussian interval), `jackknife_plus`, `cqr_norm` (normalised conformity), `cqr_proper`
(conformalised quantile regression). The conformal layers are *distribution-free* — they
give a finite-sample coverage guarantee the Gaussian-likelihood baseline lacks.

Grid = 3 base models × 4 layers, evaluated under nested LOPO-CV × N seeds.

---

## 2. New / modified files

### 2.1 Core (`src/growth/`) — formal, reusable, unit-tested

| File | Change | Public API |
|------|--------|-----------|
| `shared/growth_models.py` | **modified** | `PatientTrajectory.observation_ensemble: np.ndarray \| None` field (shape `[n_i, M]`, D=1 only, validated in `__post_init__`); `PatientTrajectory.ensemble_size` property. Backward compatible (default `None`). |
| `models/growth/ensemble_lme.py` | **new** | `EnsembleLMEGrowthModel(GrowthModel)` — Path B. `EnsembleLMEError`. `_gaussian_mixture_quantile(means, sigmas, q, weights=None)`. |
| `shared/conformal.py` | **new** | `jackknife_plus_interval(loo_predictions_at_test, loo_residuals, alpha, score)`; `SplitConformalCalibrator`; `NormalizedConformalCalibrator`; `ConformalPredictiveSystemCalibrator`; `CQRCalibrator`; `beta_binomial_coverage_ci(n_covered, n_total, confidence)`; `ConformalError`; `_jackknife_plus_ranks`. crepes-backed. |
| `evaluation/conformal_lopo.py` | **new** | `ConformalLOPOEvaluator`; `ConformalLOPOResults` (`.to_dict/.from_dict/.aggregate_metrics/.per_patient_table`); `ConformalLOPOFoldResult`; `default_cqr_features`; `ALL_LAYERS`; `ConformalLOPOError`. |
| `shared/bootstrap.py` | **modified** | added `cohen_d_paired(differences)`, `benjamini_hochberg(p_values, q)`, `paired_bootstrap_ci(values_a, values_b, ...)` (BCa, with degenerate-case fallback). `__all__` added. |
| `stages/stage1_volumetric/trajectory_loader.py` | **modified** | added `load_ensemble_trajectories_from_h5(...)` with a `scaling: "raw" \| "mean_matched"` parameter. Existing functions untouched. |
| `../../../pyproject.toml` | **modified** | added `crepes>=0.7.0` and `pyarrow>=14.0.0` to `[project.dependencies]`. Installed via `pip install -e .` in the `growth` conda env (numpy 1.26.4 unaffected). |

### 2.2 Experiment folder (`experiments/stage1_volumetric/conformal_calibration/`)

```
conformal_calibration/
├── HANDOFF.md                 # this file
├── DESIGN.md                  # the full spec — read for design intent
├── README.md                  # usage / run instructions
├── __init__.py
├── run.py                     # CLI: --write-manifest / --task-index K / --analyze / --smoke / --force
├── configs/
│   ├── picasso.yaml           # PRODUCTION: signal=men_mean_entropy, scaling=mean_matched
│   ├── picasso_variance.yaml  # ablation: signal=logvol_var, scaling=raw
│   ├── local.yaml             # full local run (same hyperparams, local H5 path)
│   └── local_smoke.yaml       # 2 seeds, 24 patients, M=5 — the smoke config
├── modules/
│   ├── __init__.py
│   ├── cohort.py              # load_cohort -> Cohort dataclass; write_cohort_meta
│   ├── runner.py              # run_task(base_model, seed, cohort, cfg, output_root)
│   ├── statistics.py          # paired BCa bootstrap + Wilcoxon + Cohen's d + BH-FDR
│   ├── aggregator.py          # collect_runs -> long-form parquet/csv
│   └── figures.py             # make_all_figures (4 figures)
├── slurm/
│   ├── launcher.sh            # writes manifest, sbatch --array, then dependent analysis job
│   ├── worker.sh              # one array task = one (base_model, seed)
│   └── analysis_worker.sh     # run.py --analyze
└── tests/
    ├── __init__.py
    └── test_config_and_manifest.py   # 29 experiment-level tests
```

### 2.3 Core unit tests (`tests/growth/`)

| File | Coverage |
|------|----------|
| `test_conformal.py` | jackknife+ ranks & coverage, split/normalised/CQR marginal coverage, normalised collapses to constant width when σ̂ uninformative, y_min/y_max clamping, beta-binomial CI. 20 tests. |
| `test_ensemble_lme.py` | law-of-total-variance exactness, degenerate ensemble == single LME, between-variance grows with member disagreement, mixture-quantile exactness, misuse errors. 12 tests. |
| `test_conformal_lopo.py` | all-layers-finite, conformal restores coverage lost by an over-confident model, serialisation round-trip, per-patient table, misuse errors. 10 tests. |

---

## 3. Architecture & key design decisions

### 3.1 The model × calibration grid
One LOPO run of a base model produces **all four** calibration layers at once (they share
the nested-LOPO residual reservoir). So the SLURM manifest is **3 base models × N seeds
tasks**, not 12 × N. `run.py --write-manifest` builds this; `--task-index K` runs task K.

### 3.2 Nested LOPO (honest evaluation)
`ConformalLOPOEvaluator` does an outer leave-one-patient-out loop; inside each fold it runs
an inner jackknife over the N−1 training patients to calibrate the conformal layers, then
predicts the held-out patient. Cost is **O(N²)** base-model fits per (model, seed) — for an
LME at N≈54 a few minutes; for `ensemble_bma` it scales ×M. SLURM-parallelised one task per
(base_model, seed). This is deliberately the rigorous nested evaluation, not the single-loop
shortcut — see `DESIGN.md §3`.

### 3.3 Protocol: `last_from_rest`, patient-level
Predict the last timepoint from all preceding ones → exactly **one residual / one interval
per patient**. Keeps patient-level exchangeability clean (no within-patient temporal
correlation leaks into the calibration set). `docs/CONFORMAL_PATH_ANALYSIS.md §4.2`.

### 3.4 Conformal backend: `crepes`
`crepes` (Boström, COPA/JMLR) operates directly on residual arrays — good impedance match
to our LOPO residuals. `ConformalRegressor.fit(residuals, sigmas=)` + `.predict_int(...)`
backs `SplitConformalCalibrator` and `NormalizedConformalCalibrator`;
`ConformalPredictiveSystem` backs the CPS calibrator. Jackknife+ aggregation (Barber et al.
2021 L/U formula) is hand-rolled — no library does the LOPO-coupled version.

### 3.5 Entropy is the default signal, mean-matched
`men_mean_entropy` (predictive entropy over MEN voxels) is **not** in (log-volume)² units.
Injecting it raw as σ²_v into `LMEHeteroGrowthModel` is a units error. `scaling: mean_matched`
multiplicatively rescales it so its cohort mean equals the cohort mean of `logvol_var`,
preserving the per-scan ranking while fixing the scale. Verified on the real H5: raw mean
0.5089 → 0.4236, exactly the `logvol_var` reference mean. This affects only `lme_hetero`'s
*parametric* layer — `cqr_norm` re-estimates the scale by construction, and `ensemble_bma`
never touches a scalar signal. `picasso.yaml` = entropy+mean_matched (default);
`picasso_variance.yaml` = logvol_var+raw (ablation).

### 3.6 Small-N robustness (the non-obvious bits)
- **CQR quantile rearrangement** (Chernozhukov et al. 2010): independently-fitted α/2 and
  1−α/2 quantile regressors can cross at small N; `CQRCalibrator` rearranges them to be
  non-crossing before conformalisation, guaranteeing `lower ≤ upper`.
- **crepes degeneracy → `y_min`/`y_max` clamping**: below ~19 calibration units a 95%
  crepes interval degenerates to "maximum size" (±inf). `ConformalLOPOEvaluator` accepts
  `y_min`/`y_max` (set from the cohort log-volume range ± margin in `runner.py`) and
  threads them to the crepes-backed layers so the interval stays finite and meaningful
  (log-volume is bounded by anatomy).
- **jackknife+ rank clamping**: `_jackknife_plus_ranks` clamps the order-statistic ranks to
  `[1, n]`; clamping (only at tiny n) widens the interval to `[min, max]` — conservative.
- **BCa degenerate fallback**: `paired_bootstrap_ci` (core) and `_bca_ci` (experiment
  `statistics.py`) fall back to the percentile interval when the bias/acceleration terms
  are degenerate (e.g. a 6-patient tertile subset where every bootstrap replicate lands on
  one side of the observed value → z₀ = Φ⁻¹(0) = −∞).

### 3.7 τ = 0 only
This experiment does **not** sweep τ. The τ-sweep null is already published; the new axis
here is the calibration layer. Each model uses the real (mean-matched) σ²_v at τ=0.
Tertile-stratified coverage/IS is retained (`docs/CONFORMAL_PATH_ANALYSIS.md §4.3`).

### 3.8 `days_from_baseline` readiness
`time.variable` switches the loaders between `timepoint_idx` (ordinal, current) and deltas
derived from `metadata/study_date`. The H5 already carries `metadata/study_date`. No code
path is ordinal-only — when real dates land, flip `time.variable: days_from_baseline` and
rerun. Smoke/production currently use `ordinal`.

### 3.9 Seeds
At τ=0 the only seed-sensitive components are (a) `LMEHeteroGrowthModel`'s multi-restart
L-BFGS-B optimiser and (b) the internal 2/3–1/3 split of `cqr_proper`. The seed axis
measures optimiser stability + CQR split variance; for `lme_homo` + {parametric,
jackknife_plus} it is near-deterministic (a reported sanity check). Statistics treat seeds
as replicates (mirrors `main_experiment`).

---

## 4. Data flow

```
MenGrowth.h5  (uncertainty/per_member_volumes [179,20], logvol_mean, logvol_var,
               men_mean_entropy, metadata/study_date, ...)
   │
   │  load_ensemble_trajectories_from_h5(scaling=...)   [trajectory_loader.py]
   ▼
PatientTrajectory per patient:
   observations         = log1p ensemble-mean log-volume  (shared y target, = logvol_mean)
   observation_variance = signal, floored (+ mean-matched if scaling=mean_matched)
   observation_ensemble = log1p(per_member_volumes)  [n_i, M]
   │
   │  load_cohort(cfg)   [modules/cohort.py]  -> Cohort (+ max_patients trunc, M trunc)
   ▼
run_task(base_model, seed, cohort, cfg, output_root)   [modules/runner.py]
   │  base_model -> {lme_homo: LMEGrowthModel,
   │                 lme_hetero: LMEHeteroGrowthModel,
   │                 ensemble_bma: EnsembleLMEGrowthModel}
   │  ConformalLOPOEvaluator(alpha, layers, ..., y_min, y_max).evaluate(model_class, trajs)
   ▼
runs/{base_model}/seed_{NNN}/{conformal_lopo_results,marginal_metrics,tertile_metrics}.json
   │
   │  run.py --analyze : aggregator.collect_runs -> statistics.run_statistics -> figures
   ▼
aggregated/{results_table.parquet, statistics.json}  +  figures/*.png
```

The H5 fact that makes Path B work: `np.log1p(uncertainty/per_member_volumes).mean(axis=1)`
equals `uncertainty/logvol_mean` exactly — the loader asserts this. So the per-member LMEs
and the baseline are centred on the same y target.

---

## 5. How to run

```bash
# fast smoke (2 seeds, 24 patients, M=5; a few minutes; entropy config)
python -m experiments.stage1_volumetric.conformal_calibration.run \
    --config experiments/stage1_volumetric/conformal_calibration/configs/local_smoke.yaml --smoke

# full local run (all patients, 20 seeds)
python -m experiments.stage1_volumetric.conformal_calibration.run \
    --config experiments/stage1_volumetric/conformal_calibration/configs/local.yaml --smoke

# Picasso: launcher writes the manifest, submits the array, then the dependent analysis job
bash experiments/stage1_volumetric/conformal_calibration/slurm/launcher.sh \
    experiments/stage1_volumetric/conformal_calibration/configs/picasso.yaml

# tests
python -m pytest tests/growth/test_conformal.py tests/growth/test_ensemble_lme.py \
    tests/growth/test_conformal_lopo.py experiments/stage1_volumetric/conformal_calibration/tests/ -q
```

`run.py` is always invoked as a module (`python -m ...`), never `python run.py` — it uses
package-relative imports, mirroring `main_experiment`.

---

## 6. Verification status (as of 2026-05-14)

| Check | Result |
|-------|--------|
| Core unit tests (`tests/growth/test_{conformal,ensemble_lme,conformal_lopo}.py`) | **42/42 pass** |
| Experiment tests (`tests/test_config_and_manifest.py`) | **29/29 pass** |
| Regression suite (`pytest -m "phase4 or evaluation"`, 316 tests) | **309 passed, 7 skipped, 0 failed** |
| Local smoke — variance config, full `--smoke` | exit 0, all 6 tasks, analysis OK |
| Local smoke — entropy config, full `--smoke` | exit 0, 0 errors; `results_table.parquet`, `statistics.json` (bootstrap CIs + Wilcoxon + Cohen's d + BH-FDR), 4 figures |
| `mean_matched` path on real H5 | verified: entropy mean 0.5089 → 0.4236 == logvol_var ref mean |

Smoke numbers are an N=24 subset — **not** the result. They confirm the pipeline runs
end-to-end and every layer produces finite metrics for every base model.

---

## 7. Open items / next steps

### Must do before the Picasso run
1. **Sync the merged H5 to Picasso.** `picasso.yaml` / `picasso_variance.yaml` point at
   `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/MenGrowth.h5`
   (the established Picasso dataset path). The merged file with `uncertainty/per_member_volumes`
   currently lives locally at `/media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5`.
   An older Picasso copy may lack the `uncertainty` group entirely or the
   `per_member_volumes` dataset — `ensemble_bma` and the loader's consistency assertion
   will fail without it. rsync the merged file first.

### Deferred / known minor issues
- **`pit_grid` figure dropped.** PIT needs a predictive *distribution*; it is not defined
  for interval-only conformal layers (jackknife+, cqr_*). If wanted, implement it for the
  `parametric` layer only (Gaussian PIT via `growth.shared.metrics.compute_pit`) — or use
  `ConformalPredictiveSystemCalibrator`, which does give a full predictive distribution and
  a distribution-free CRPS. The 4 retained figures carry the IS / coverage / tertile /
  width-vs-σ²_v story.
- **`ConformalLOPOResults.model_name` shows `LME(D=0, ...)`.** Cosmetic: `model_name` is
  read from `model_class(**kwargs).name()` on an *unfitted* instance, and
  `LMEGrowthModel._obs_dim` is 0 until `fit()`. Does not affect any computation; the saved
  results carry the slightly-wrong label. Fix would be to set the label after the first
  fit inside `ConformalLOPOEvaluator`.
- **`statistics.py` reimplements BCa** (`_bca_ci`) rather than calling
  `growth.shared.bootstrap.paired_bootstrap_ci`. It was patched to add the same
  degenerate-case fallback, so it is correct, but consolidating onto the core function
  would remove the duplication.

### Scientific extensions noted in the parent doc (not implemented here)
- **Path C — cross-segmenter disagreement** (`docs/CONFORMAL_PATH_ANALYSIS.md §6`): variance
  of log-volume *across* segmenters (BSF-LoRA / BraTS25 / BraTS23-GLI). A different
  epistemic axis from within-LoRA variance; would plug in as one more `uncertainty.signal`.
- **CQR-d** (density-calibrated): explicitly out of scope at N=54 (`§3.3c`, `§9`).

---

## 8. Interface reference (for wiring against)

```python
# growth.shared.growth_models
PatientTrajectory(patient_id, times, observations, covariates=None,
                  observation_variance=None, observation_ensemble=None)   # ensemble: [n_i, M]

# growth.models.growth.ensemble_lme
EnsembleLMEGrowthModel(method="reml", n_members=None, use_covariates=False,
                       covariate_names=None, missing_strategy="skip")     # GrowthModel ABC

# growth.shared.conformal
jackknife_plus_interval(loo_predictions_at_test, loo_residuals, alpha=0.05,
                        score="signed") -> (lower, upper)
SplitConformalCalibrator(confidence=0.95).fit(residuals).predict_interval(y_hat, y_min, y_max)
NormalizedConformalCalibrator(confidence=0.95, sigma_floor=1e-6)
    .fit(residuals, sigmas).predict_interval(y_hat, sigma_hat, y_min, y_max)
CQRCalibrator(alpha=0.05, calib_fraction=0.33, quantile_reg_alpha=0.0,
              solver="highs", seed=42).fit(x, y).predict_interval(x)
beta_binomial_coverage_ci(n_covered, n_total, confidence=0.95) -> (lo, hi)

# growth.evaluation.conformal_lopo
ConformalLOPOEvaluator(alpha=0.05, layers=("parametric","jackknife_plus","cqr_norm","cqr_proper"),
    jackknife_score="signed", cqr_calib_fraction=0.33, cqr_feature_fn=None, seed=42,
    y_min=-inf, y_max=inf).evaluate(model_class, patients, **model_kwargs) -> ConformalLOPOResults
# ConformalLOPOResults.aggregate_metrics() keys: "r2_log", "{layer}/is_95",
#   "{layer}/coverage_95", "{layer}/coverage_95_ci_{low,high}", "{layer}/mean_width"
# ConformalLOPOResults.per_patient_table() -> long-form rows (patient_id, layer, actual,
#   lower, upper, width, covered, interval_score, sigma_v_sq_target, ...)

# growth.shared.bootstrap
cohen_d_paired(differences) -> float
benjamini_hochberg(p_values, q=0.05) -> (rejected, p_adjusted)
paired_bootstrap_ci(values_a, values_b, n_bootstrap=10000, confidence_level=0.95, seed=42)
    -> BootstrapResult(estimate, ci_lower, ci_upper, ...)   # estimate = mean(a) - mean(b)

# growth.stages.stage1_volumetric.trajectory_loader
load_ensemble_trajectories_from_h5(h5_path, *, time_variable="ordinal",
    variance_key="logvol_var", mean_key="logvol_mean", scaling="raw",
    floor_variance=1e-6, exclude=None, min_timepoints=2, skip_all_zero_volume=True,
    max_logvol_std=None, missing_date_strategy="skip") -> list[PatientTrajectory]
```

---

## 9. References

- Barber, Candès, Ramdas, Tibshirani. *Predictive inference with the jackknife+.*
  Annals of Statistics 49(1):486–507, 2021.
- Romano, Patterson, Candès. *Conformalized quantile regression.* NeurIPS 2019.
- Papadopoulos, Vovk, Gammerman. *Normalized nonconformity measures for regression
  conformal prediction.* AIAI 2008.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-free predictive inference
  for regression.* JASA 113(523):1094–1111, 2018.
- Chernozhukov, Fernández-Val, Galichon. *Quantile and probability curves without
  crossing.* Econometrica 78(3):1093–1125, 2010.
- Boström. *crepes: a Python package for conformal regressors and predictive systems.*
  COPA 2022.
</content>
</invoke>
