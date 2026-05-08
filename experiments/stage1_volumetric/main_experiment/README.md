# Main Experiment — σ²_v Propagation under a τ-Shift Sweep

## 1. Question

> Does propagating per-scan segmentation uncertainty σ²_v through a
> heteroscedastic LME shift the predictive distribution toward where it
> should be — i.e. produce better calibration (IS@95, cov@95, sharpness) —
> than a homoscedastic LME that assumes a constant per-observation variance?

## 2. Design

* **Cohort:** 54 patients / 163 scans, QC-filtered with `max_logvol_std ≤ 1.0`.
* **Models:** LME (homoscedastic, σ²_v ignored), `LMEHetero@σ²_v=floor`
  (zero-injection control to isolate the structural REML/EBLUP advantage),
  `LMEHetero@injected` (varies per cell).
* **Validation:** LOPO-CV with `last_from_rest`.
* **σ²_v generator (τ-shift sweep):** bootstrap the empirical post-QC
  log σ²_v vector and add a global log-space shift τ:
  $$\sigma^2_{v,k}(\tau) \;=\; \exp\bigl(L_k + \tau\bigr),\;\; L_k \stackrel{\mathrm{iid}}{\sim} \widehat{F}_{\log\sigma^2_v}^{\text{post-QC}}.$$
  Shape is preserved exactly; τ=0 is the empirical-match cell *by construction*.
  Saturated extremes are clipped to floor (1e-3) and ceiling (50.0, ≈ signal-variance scale).
* **Primary sweep grid:** τ ∈ {−7.11, −4.61, −2.12, 0.00, +2.86, +5.35,
  +7.84, +10.33, +12.82} — 9 levels anchored at p5/p95 of empirical log σ²_v
  with a 2-log-unit safety margin.
* **Seeds:** R=20 production / R=2 smoke.
* **Ablation (optional):** Beta(α) sweep, α ∈ {−1.0, 0.0, +1.0}, R=10.

## 3. Falsifiable Predictions (committed before unblinding)

1. At τ=τ_min (saturated low) `LMEHetero@injected ≈ LMEHetero@floor ≈ LME` —
   three-way convergence. *If not, the structural REML advantage is dominant.*
2. At τ=τ_max (saturated high), cov@95 → 1, IS@95 grows (over-conservative);
   ΔIS@95 turns positive and unbounded.
3. Spearman ρ(τ, ci_width) > 0 (sharpness scales with injected noise).
4. R²_log roughly constant across τ (point predictions barely depend on σ²_v
   at non-extreme τ).
5. Marginal cov@95 stays near nominal 0.95 in a neighbourhood of τ=0.

## 4. Statistical Tests (committed up front)

| Test | Scope | Spec |
|------|-------|------|
| Paired bootstrap of ΔIS@95, ΔR², Δcov@95 | Marginal + per σ²_v tertile | B=10,000, BCa CI, two-sided p |
| Wilcoxon signed-rank + Cohen's d | Per-patient ΔIS@95 | Paired by patient_id |
| Spearman ρ | Across-τ trend | τ vs median across-seed Δmetric |
| KS-vs-Uniform on PIT | Per cell | distributional calibration |
| Benjamini–Hochberg FDR | Family of p-values | q=0.05; separate FDR families for marginal vs tertile |

## 5. Reproducing the Experiment

### Local smoke (~10 min)

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

### Picasso (full sweep)

```bash
bash experiments/stage1_volumetric/main_experiment/slurm/launcher.sh \
  experiments/stage1_volumetric/main_experiment/configs/picasso.yaml --dry-run
# inspect, then re-run without --dry-run
```

The launcher:
1. Activates the configured conda env (default `mengrowth`).
2. Builds the manifest (`manifest.json`) — 9 τ × 20 seeds + 2 baselines = 182 tasks.
3. Submits a SLURM array job (`--array=0-181%THROTTLE`) of `worker.sh`.
4. Submits an analysis job with `--dependency=afterany` of the array.

If the analysis hits its time limit (B=10000 bootstrap is heavy across
360 contrasts, ≈ 3 h), re-submit just the analysis with extended time:

```bash
sbatch --time=0-04:00:00 \
    --export=ALL,CONFIG_PATH=experiments/stage1_volumetric/main_experiment/configs/picasso.yaml,CONDA_ENV=mengrowth,REPO_DIR=<REPO_DIR> \
    experiments/stage1_volumetric/main_experiment/slurm/analysis_worker.sh
```
or drop `statistics.bootstrap.n_samples: 2000` in `picasso.yaml` (≈ 5× faster).

## 6. Output Layout

```
{output_dir}/
├── tau_grid.json                       # resolved τ-grid + saturation parameters
├── cohort_meta.json                    # patient ids, n_timepoints, σ²_v vector
├── manifest.json                       # ordered list of array tasks
├── LME_baseline/
│   ├── lopo_results.json
│   ├── marginal_metrics.json
│   └── tertile_metrics.json
├── LMEHetero_Zero_baseline/
│   └── …
├── runs/
│   ├── empirical_shift_tau_+0.000/seed_NNN/
│   │   ├── sigma_v_sq_injected.npy
│   │   ├── lopo_results.json
│   │   ├── marginal_metrics.json
│   │   └── tertile_metrics.json
│   ├── empirical_shift_tau_*.*/…
│   └── beta_alpha_*/…                  # ablation (if enabled)
├── aggregated/
│   ├── results_table.{parquet,csv}     # long-form (family, level, seed, scope, tertile, metric, value)
│   ├── bootstrap_results.json          # B=10,000 paired bootstrap + BH-adjusted p
│   ├── wilcoxon_results.json
│   └── spearman_results.json
├── figures/
│   ├── cov_vs_tau.png
│   ├── is_vs_tau.png
│   ├── sharpness_panel.png
│   └── pit_grid.png
└── data/
    ├── figures/tau_sweep_surface.{png,pdf}    # 3-D σ²_v sweep visualisation
    ├── tau_sweep_surface_data.npz
    └── tau_sweep_surface_metadata.json
```

## 7. Results Snapshot (run 2026-05-07, 9 τ × 20 seeds)

LME homo: IS@95 = 8.286, cov95 = 0.907, NLPD = 1.78. Median across seeds:

| τ                  | IS@95  | cov95 | sharp | NLPD | % above min IS |
|--------------------|------:|------:|------:|-----:|---------------:|
| −7.11 (sat. low)   | 8.287 | 0.907 |  4.42 | 1.78 |     +4.5 % |
| **0.00 (empirical)** | **8.248** | **0.907** | **4.43** | **1.79** | **+4.0 %** |
| **+2.86 (optimum)** | **7.931** | **0.926** | **4.92** | **1.69** | **0.0 %** |
| +5.35              | 8.971 | 0.944 |  7.47 | 1.78 |    +13.1 % |
| +7.84              | 13.853| 1.000 | 13.85 | 2.13 |    +74.7 % |
| +12.82 (sat. high) | 27.928| 1.000 | 27.93 | 2.90 |   +252.1 % |

IQR across seeds at τ=0: IS@95 ∈ [8.08, 8.37] — overlaps homo (8.286).

**Bootstrap (B=10,000, BCa CI; LME → LMEHetero@injected; median across 20 seeds):**

| τ                  | ΔIS_med | CI95             | p_med  | BH-rej / 20 |
|--------------------|---:|:---:|---:|---:|
| −7.11 (sat. low)   | +0.001 | [−0.000, +0.003] | 0.52 | 0/20 |
| 0.00 (empirical)   | −0.038 | [−0.30, +0.23]   | 0.58 | 0/20 |
| +2.86 (point min)  | −0.355 | [−2.36, +1.57]   | 0.35 | 1/20 |
| +5.35              | +0.685 | [−4.23, +4.47]   | 0.70 | 4/20 |
| +7.84              | +5.57  | [−0.17, +10.0]   | 0.06 | 0/20 |
| +10.33             | +16.6  | [+11.0, +20.4]   | <1e-4 | **20/20** |
| +12.82 (sat. high) | +19.6  | [+14.2, +23.3]   | <1e-4 | **20/20** |

LMEHetero_Zero → LMEHetero_Injected reproduces these numbers to within 1e-3
on every τ → the structural REML/EBLUP advantage is zero on this cohort.

**Key findings (post-bootstrap):**

1. **The empirical σ²_v is statistically indistinguishable from LME homo.**
   ΔIS@95 = −0.04 with CI [−0.30, +0.23] at τ=0; 0/20 BH-rejected. The 0.5 %
   point-estimate edge is firmly within seed noise.
2. **The IS minimum at τ=+2.86 is NOT significant.** Bootstrap CI [−2.36, +1.57]
   straddles zero (p=0.35). Earlier "near-optimal" framing was a point-estimate
   artefact and is retracted.
3. **The only robust effect is degradation under over-estimation.** Bootstrap
   rejects ΔIS=0 only at τ ≥ +10.33 (20/20 BH). Below that, ΔIS CIs straddle
   zero everywhere — including the high-σ²_v tertile (ΔIS=+0.02, CI [−0.36,
   +0.12] at τ=0).
4. **The asymmetric Gneiting–Raftery loss is the mechanism.** Once cov95 → 1
   the miscoverage term vanishes and IS reduces to interval width, which scales
   ∝ e^(τ/2): IS keeps growing without bound, so over-estimation is much
   costlier than under-estimation (which collapses to homo).
5. **Hetero is "safe but unhelpful" on this cohort.** Injecting the empirical
   LoRA σ²_v does not degrade calibration anywhere in τ ∈ [−7, +5] but does
   not improve it either. To beat homo would require an ensemble with σ²_v
   roughly an order of magnitude more dispersed than M=20 produces, or larger
   N to tighten the bootstrap CIs.
6. **The historical high-tertile win does NOT replicate** in this controlled
   contrast (LMEHetero@injected vs LMEHetero@floor as the structural-effect
   control). The prior result likely conflated the structural REML variance
   advantage with σ²_v propagation; with the control here, only propagation
   remains and it fails to reach significance.

## 8. Time Variable Note

The current config uses `time.variable: ordinal` because per-scan study
dates are not yet ingested into MenGrowth.h5. The pipeline supports
`days_from_baseline`; once dates are available, change one config line
and re-submit. Long-horizon claims await real timestamps.
