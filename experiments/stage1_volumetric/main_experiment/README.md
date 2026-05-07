# Main Experiment — σ²_v Propagation under a Controlled π-Sweep

## 1. Question

> Does propagating per-scan segmentation uncertainty σ²_v through a
> heteroscedastic LME shift the predictive distribution toward where it
> should be — i.e. produce better calibration (IS@95, cov@95, sharpness) —
> than a homoscedastic LME that assumes a constant per-observation variance?

## 2. Design

* **Cohort:** 54 patients / 163 scans, QC-filtered with `max_logvol_std=1.0`.
* **Models:** LME (homoscedastic, σ²_v ignored), `LMEHetero@σ²_v=floor` (zero-injection
  control to isolate the structural REML/EBLUP advantage), `LMEHetero@injected`
  (varies per cell).
* **Validation:** LOPO-CV with `last_from_rest`.
* **σ²_v generator:** 2-component log-normal mixture, EM-fit to the **pre-QC**
  empirical σ²_v vector (179 scans, includes the high-noise tail). Frozen
  components $(\mu_L, \sigma_L, \mu_H, \sigma_H)$; the mixture weight π is the
  only knob.
* **Primary sweep grid:** π ∈ {0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75,
  0.875, 1.00} (9 levels) **+ empirical pass-through** (sanity-match cell at
  π = π̂).
* **Seeds:** R=20 production / R=2 smoke.
* **Ablation:** Beta(α) sweep, α ∈ {-1.0, 0.0, +1.0}, R=10.

## 3. Falsifiable Predictions (committed before unblinding)

1. At π = π̂ the parametric mixture cell matches the empirical pass-through
   within bootstrap CI. *If not, the mixture fit is unfaithful.*
2. At π=1 (all clean), `LMEHetero@injected ≈ LMEHetero@0 ≈ LME` — three-way
   convergence. *If not, the structural REML advantage is dominant.*
3. At π=0 (all noisy), cov@95 → 1, IS@95 grows (over-conservative);
   ΔIS@95 turns positive.
4. Spearman ρ(π, ΔIS@95_high_tertile) > 0: more clean scans = less rescue
   room, ΔIS@95 shrinks toward zero.
5. R²_log roughly constant across π (point predictions barely depend on σ²_v).
6. Marginal cov@95 stays near nominal 0.95 across the sweep.

## 4. Statistical Tests (committed up front)

| Test | Scope | Spec |
|------|-------|------|
| Paired bootstrap of ΔIS@95, ΔR², Δcov@95 | Marginal + per σ²_v tertile | B=10,000, BCa CI, two-sided p |
| Wilcoxon signed-rank + Cohen's d | Per-patient ΔIS@95 | Paired by patient_id |
| Spearman ρ | Across-π trend | π vs median across-seed Δmetric |
| KS-vs-Uniform on PIT | Per cell | distributional calibration |
| Benjamini–Hochberg FDR | Family of p-values | q=0.05; separate FDR families for marginal vs tertile |

## 5. Reproducing the Experiment

### Local smoke (≤ 30 min)

```bash
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.main_experiment.run \
  --config experiments/stage1_volumetric/main_experiment/configs/local_smoke.yaml \
  --smoke
```

### Single cell

```bash
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.main_experiment.run \
  --config <CONFIG.yaml> --task-index <K>
```

### Picasso (12 h budget)

```bash
bash experiments/stage1_volumetric/main_experiment/slurm/launcher.sh \
  experiments/stage1_volumetric/main_experiment/configs/picasso.yaml --dry-run
# inspect, then re-run without --dry-run
```

The launcher:
1. Builds the manifest (`manifest.json`) with one entry per (model | π | seed).
2. Submits a SLURM array job (`--array=0-N%THROTTLE`) of `worker.sh`.
3. Submits an analysis job with `--dependency=afterany` of the array.

## 6. Output Layout

```
{output_dir}/
├── mixture_fit.json                     # frozen 2-comp LogNormal mixture
├── cohort_meta.json                     # patient ids, n_timepoints, σ²_v vector
├── manifest.json                        # ordered list of array tasks
├── LME_baseline/
│   ├── lopo_results.json                # cached, σ²_v-independent
│   ├── marginal_metrics.json
│   └── tertile_metrics.json
├── LMEHetero_Zero_baseline/
│   └── ...
├── runs/
│   ├── lognormal_mixture_pi_0.000/seed_NNN/
│   │   ├── sigma_v_sq_injected.npy
│   │   ├── lopo_results.json
│   │   ├── marginal_metrics.json
│   │   └── tertile_metrics.json
│   ├── lognormal_mixture_pi_*.*/...
│   ├── empirical_emp/seed_000/...
│   └── beta_alpha_*/...                 # ablation
├── aggregated/
│   ├── results_table.{parquet,csv}      # long-form (family, level, seed, scope, tertile, metric, value)
│   ├── bootstrap_results.json           # B=10,000 paired bootstrap + BH-adjusted p
│   ├── wilcoxon_results.json
│   └── spearman_results.json
└── figures/
    ├── cov_vs_pi.png
    ├── is_vs_pi.png
    ├── sharpness_panel.png
    └── pit_grid.png
```

## 7. Pre-flight Validation

Before submitting to Picasso, the following must hold:

- [x] Mixture fit: π̂ ∈ [0.7, 0.95] and KS p ≥ 0.05 vs empirical log σ²_v.
- [x] All 16 unit tests pass: `pytest experiments/stage1_volumetric/main_experiment/tests/`.
- [x] Local smoke completes 9 cells + analysis < 10 min.
- [x] All four figures render without errors.
- [x] `aggregated/results_table.csv` is non-empty with the expected columns.
- [ ] Picasso paths in `configs/picasso.yaml` exist (run via `pre-flight` skill).
- [ ] Conda env `mengrowth` available on Picasso login node.

## 8. Time Variable Note

The current config uses `time.variable: ordinal` because per-scan study dates
are not yet ingested into MenGrowth.h5. The pipeline already supports
`days_from_baseline`. Once dates are available, change one config line and
re-submit; nothing else needs updating.

UQ_THESIS_GAP_ANALYSIS §3.1 flags ordinal time as a limitation for
long-horizon claims — this experiment focuses on calibration shape along the
π-sweep, where ordinal is sufficient.
