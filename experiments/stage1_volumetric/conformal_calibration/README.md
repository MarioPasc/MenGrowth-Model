# Conformal Calibration Experiment

Evaluates four conformal calibration layers (parametric, jackknife+, CQR-normalised, CQR-proper)
on three base growth models (LME-homo, LME-hetero, Ensemble-BMA) under LOPO-CV.

## Structure

```
conformal_calibration/
  run.py                      # CLI entry point
  modules/
    cohort.py                 # Cohort loading with ensemble + variance fields
    runner.py                 # Task execution: (base_model, seed) → JSON metrics
    aggregator.py             # Walk runs/ → long-form Pandas DataFrame
    statistics.py             # BCa bootstrap, Wilcoxon, BH-FDR
    figures.py                # Matplotlib figures from aggregated table
  configs/
    local_smoke.yaml          # 2 seeds, 24 patients, M=5  (a few min local)
    local.yaml                # Full local run
    picasso.yaml              # Picasso config — signal = men_mean_entropy (mean-matched)
    picasso_variance.yaml     # Ablation — signal = logvol_var (raw)
  slurm/
    launcher.sh               # Submit array + dependent analysis job
    worker.sh                 # Per-task SLURM worker
    analysis_worker.sh        # Aggregation + figures worker
  tests/
    test_config_and_manifest.py
```

## Quick start

Smoke run (all tasks sequentially, no real H5 required for tests):

```bash
~/.conda/envs/growth/bin/python -m pytest \
    experiments/stage1_volumetric/conformal_calibration/tests/ -q
```

Write manifest only:

```bash
~/.conda/envs/growth/bin/python \
    experiments/stage1_volumetric/conformal_calibration/run.py \
    --config experiments/stage1_volumetric/conformal_calibration/configs/local_smoke.yaml \
    --write-manifest
```

## SLURM submission (Picasso)

```bash
bash experiments/stage1_volumetric/conformal_calibration/slurm/launcher.sh \
    experiments/stage1_volumetric/conformal_calibration/configs/picasso.yaml
```

Dry-run:

```bash
bash experiments/stage1_volumetric/conformal_calibration/slurm/launcher.sh \
    experiments/stage1_volumetric/conformal_calibration/configs/picasso.yaml --dry-run
```

## Output layout

```
{output_dir}/
  cohort_meta.json
  manifest.json
  runs/
    {base_model}/
      seed_{NNN}/
        conformal_lopo_results.json
        marginal_metrics.json
        tertile_metrics.json
  aggregated/
    results_table.parquet   (or .csv fallback)
    statistics.json
  figures/
    is_by_model_calibration.png
    coverage_by_model_calibration.png
    tertile_panel.png
    width_vs_sigmav.png
```

## Key design decisions

- One SLURM task = one (base_model, seed) pair. The manifest decouples
  task enumeration from job submission, enabling clean array indexing.
- Tertile cuts are pinned to the full cohort empirical σ²_v distribution
  (computed once from the cohort, passed into all metric helpers) to prevent
  fold-to-fold leakage.
- `--constraint=cpu` is passed explicitly on every `sbatch` call in
  `launcher.sh` to work around the Picasso lua plugin which overrides
  `#SBATCH --constraint` directives.
