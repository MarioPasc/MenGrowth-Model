# Diagnostic — Candidate Segmentation Uncertainty Signals as σ²_v

A complement to `experiments/stage1_volumetric/main_experiment/`. Tests whether
any **alternative per-scan summary statistic** of the M=20 LoRA ensemble carries
more useful information for LME-hetero propagation than the current
`logvol_std²`. Same N=54 cohort and LOPO-CV protocol as the main experiment, so
results compare directly against `LME_baseline/lopo_results.json`.

## Question

Main-experiment τ-sweep showed ΔIS@95 ≈ 0 between LME-homo and LME-hetero,
even at extreme σ²_v scales. Two competing explanations:

1. **Wrong summary statistic** — ensemble log-volume variance throws away
   spatial / boundary information that *does* predict trajectory residuals.
2. **Information-poor signal** — the LoRA-ensemble is uninformative about the
   biological growth residual at any aggregation level.

This diagnostic distinguishes them: if a candidate beats `logvol_var` under
the same cohort and LOPO protocol, hypothesis 1 wins. If none does, the
information cost of switching to LMEHetero is unrecoverable on this cohort.

## Two-stage protocol

* **Stage 1** (cheap) — for each candidate $c_k$, correlate against the
  homoscedastic LOPO residual $|y_k - \hat\mu_k^\text{homo}|$ from
  `main_experiment/LME_baseline/lopo_results.json`. Pearson, Spearman,
  Kendall + percentile bootstrap CIs. R² of the linear fit.

* **Stage 2** (full) — run LME-hetero LOPO with $\sigma^2_v$ replaced by
  the candidate vector × scaling. Compute mean IS@95 + sharpness + miss +
  cov95, plus paired BCa-style bootstrap of ΔIS vs LME-homo with BH-FDR
  correction across all (candidate, scaling) cells.

## Candidates (`modules/candidates.py`)

11 per-scan signals from the H5 `/uncertainty/` group, plus 3 negative
controls and a sanity-check LME-homo re-run:

| name                              | family               | source |
|-----------------------------------|----------------------|--------|
| `logvol_var`                      | epistemic, scalar    | `logvol_std²` (= main-experiment baseline) |
| `logvol_mad_var`                  | epistemic, robust    | `logvol_mad_scaled²` |
| `vol_cv2`                         | relative             | $(\sigma_V/\mu_V)^2$ |
| `mean_entropy`                    | total, voxel-avg     | `mean_entropy` (post Stage-0 repair) |
| `mean_mi`                         | epistemic (BALD)     | `mean_mi` (post Stage-0 repair) |
| `mean_var_voxel`                  | epistemic, voxel-avg | `mean_var` |
| `men_entropy`                     | total, MEN region    | `men_mean_entropy` (post Stage-0 repair) |
| `men_mi`                          | epistemic, MEN       | `men_mean_mi` (post Stage-0 repair) |
| `men_boundary_entropy`            | total, boundary band | `men_boundary_entropy` |
| `men_boundary_mi`                 | epistemic, boundary  | `men_boundary_mi` (partial; needs repair) |
| `composite_logvol_x_boundary_entropy` | composite        | $\sigma^2_{\log V}\cdot(1+\beta H_{\text{boundary, MEN}})$ |

Controls: `zero`, `constant_mean`, `permuted` (shuffled empirical σ²_v).
Each candidate is run under two scalings: **`raw`** (use as-is) and
**`mean_matched`** (rescale to share cohort mean with `logvol_var`).

## Stage 0 — H5 uncertainty group is partially broken

Direct inspection of `MenGrowth.h5` showed the following are NaN-heavy
and need re-aggregation:

| dataset | NaN count / 179 | comment |
|---|--:|---|
| `mean_entropy`, `men_mean_entropy` | 163 | numerical NaN (likely `log 0` without ε) |
| `mean_mi`, `men_mean_mi` | 176 | also uniformly clamped to 0 where finite |
| `men_boundary_mi` | 48 | partial |

Per-member soft-probability NIfTIs are saved on the Sandisk2TB volume at
`uncertainty_segmentation/.../r32_M20_s42/predictions/{patient_id}-{tp:03d}/`,
but **only 5 / 179** scan folders contain `ensemble_probs.nii.gz` +
`member_*_probs.nii.gz` (the rest contain only the hard ensemble mask).

### What this means

* **Local repair** (`recompute_h5_uncertainty.py`) successfully re-aggregates
  the broken scalars for the 5 scans with full data on CPU.
* **Full repair** (174 missing scans) requires GPU re-inference. The
  `repair_*` SLURM scripts assume the predictions live at the Picasso path
  declared in `configs/picasso.yaml` — **adjust `stage0.predictions_root` if
  the per-member probs are not yet synced to Picasso**, or extend the repair
  script to call `EnsemblePredictor.predict_scan` on missing scans.

The diagnostic itself runs cleanly on the *valid* candidates without Stage 0.

## File layout

```
configs/                          local_smoke.yaml + picasso.yaml
extract_candidates.py             H5 → candidate_signals.csv (one row per scan)
recompute_h5_uncertainty.py       Stage 0: per-scan re-aggregation from per-member NIfTIs
patch_h5_uncertainty.py           Stage 0: CSV → in-place H5 patch (with timestamped backup)
run.py                            Stage 2 entry: --task-index K runs one (candidate, scaling)
modules/
  candidates.py                   registry + scaling transforms
  diagnostic.py                   Stage 1 correlation logic
  runner.py                       Stage 2 task dispatch (mirrors main_experiment.runner)
analyses/
  stage1_correlations.py          Stage 1 driver
  aggregate_stage2.py             Stage 2 aggregation + paired bootstrap + BH-FDR
  plot_stage1_correlation.py      Forest plot of Spearman ρ per candidate
  plot_stage2_is_per_candidate.py Two-panel ΔIS forest + sharpness/miss decomposition
slurm/
  repair_launcher.sh              Stage 0 array launcher (179 scans + dependent patch)
  repair_worker.sh                Stage 0 worker (per-task CSV)
  repair_patch.sh                 Stage 0 patch (concat per-task CSVs, write H5)
  launcher.sh                     Stage 2 array launcher (26 tasks, optional --depend-on)
  worker.sh                       Stage 2 worker (one task)
  analysis_worker.sh              Post-array: Stage 1 + Stage 2 analyses + figures
```

Output root (Sandisk2TB):

```
results/uncertainty_propagation_volume_prediction/test_segmentation_uncertainty_signals/
  candidate_signals.csv           179 rows × ~16 cols
  recomputed_uncertainty.csv      Stage 0 sidecar
  stage1_diagnostic/              correlations.csv + correlations.json
  runs/candidate_<name>_scaling_<raw|mean_matched>/
    {sigma_v_sq_injected.npy, lopo_results.json, marginal_metrics.json, tertile_metrics.json}
  aggregated/
    candidate_ranking.csv, bootstrap_paired_BCa.json, bh_fdr_results.json
  figures/
    stage1_correlation_panel.{pdf,png}, stage2_is_per_candidate.{pdf,png}
```

## Local smoke test (≤ 5 min total)

```bash
cd /home/mpascual/research/code/MenGrowth-Model
CFG=experiments/stage1_volumetric/test_candidate_uncertainty_signals/configs/local_smoke.yaml

# Stage 1 prep
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.extract_candidates --config $CFG

# Stage 0 smoke (2 scans — only the first will have full data)
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.recompute_h5_uncertainty --config $CFG --scan-indices 0,1
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.patch_h5_uncertainty \
    --csv /tmp/test_candidate_uncertainty_signals_smoke/recomputed_uncertainty.csv \
    --h5 /media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final/h5_format/MenGrowth.h5 \
    --dry-run

# Stage 1 correlations
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.stage1_correlations --config $CFG

# Stage 2 — single task on full N=54 cohort (≈ 30 s for LME homo, 60 s for LMEHetero)
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run --config $CFG --task-index 3 --force   # homo_sanity
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.run --config $CFG --task-index 0 --force   # logvol_var × mean_matched

# Aggregate + figures
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.aggregate_stage2 --config $CFG
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.plot_stage1_correlation --config $CFG
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.test_candidate_uncertainty_signals.analyses.plot_stage2_is_per_candidate --config $CFG
```

Sanity check: `homo_sanity` should produce IS@95 ≈ 8.29 (the published
`LME_baseline` value). Smoke run reproduces this within numerical tolerance.

## Picasso run

```bash
# Phase A — H5 repair (only if per-member probs are mirrored to Picasso)
bash experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/repair_launcher.sh \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/configs/picasso.yaml
#  -> note the patch job ID printed at the end

# Phase B — Stage 2 sweep (waits on the repair patch via --depend-on)
bash experiments/stage1_volumetric/test_candidate_uncertainty_signals/slurm/launcher.sh \
    experiments/stage1_volumetric/test_candidate_uncertainty_signals/configs/picasso.yaml \
    --depend-on <REPAIR_PATCH_JOB_ID>
```

If Phase A is skipped, the broken candidates (`mean_entropy`, `mean_mi`,
`men_entropy`, `men_mi`, `men_boundary_mi`) will run with zero-filled NaNs
and should be excluded from the thesis claim. Document this in the
methodology chapter.

## Acceptance — verified by smoke test (2026-05-08)

* Imports: all modules load cleanly (`growth` env, Python 3.11).
* Stage 0 dry-run patch: prints planned overwrites; writes nothing.
* Stage 1 diagnostic: 54 paired residuals, 2 candidates, Spearman + CI computed.
* Stage 2 (`task-index 3` = homo_sanity): IS@95 = 8.286, cov95 = 0.907 — matches main_experiment.
* Stage 2 (`task-index 0` = logvol_var × mean_matched): IS@95 = 8.478, cov95 = 0.907.
* Aggregator + BH-FDR: writes `candidate_ranking.csv`, `bootstrap_paired_BCa.json`, `bh_fdr_results.json`.
* Figures: `stage1_correlation_panel.{pdf,png}` and `stage2_is_per_candidate.{pdf,png}` render without warnings.

## Next steps before the thesis

1. Decide whether to launch the GPU re-inference for the 174 partial scans
   (≈ 5–10 h on Picasso A100) or to drop the entropy/MI candidates from
   the thesis claim with explicit framing.
2. Sync `recomputed_uncertainty.csv` (and the predictions tree if Phase A
   runs on Picasso) to the cluster, then submit the launchers above.
3. The Stage 1 diagnostic is the **fastest screen**: if no candidate beats
   $|\rho| \approx 0.1$ in the 95 % CI, the thesis conclusion ("segmentation
   uncertainty is information-poor for trajectory residuals on this cohort")
   is already supported and Stage 2 confirms downstream.
