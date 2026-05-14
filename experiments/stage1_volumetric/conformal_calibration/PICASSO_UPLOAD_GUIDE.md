# Conformal Calibration — Picasso Submission Guide

**Date:** 2026-05-14
**Status:** Ready to submit, pending the 3 prerequisites in §2.
**Prepared by:** per-patient persistence + aggregation + headline figure added; SLURM
scripts reviewed; tests executed (§5).

This guide is the pick-up point for launching `conformal_calibration` on Picasso.
Read §1 first, satisfy §2, then follow §3.

---

## 1. What you must have read before launching

| Document | Why |
|----------|-----|
| `DESIGN.md` | The comparison grid (3 base models × 4 calibration layers), nested-LOPO protocol, config schema. |
| `HANDOFF.md` | What was built, the data-flow diagram, the τ=0 design decision, the open items. |
| `../../../docs/CONFORMAL_PATH_ANALYSIS.md` | Scientific rationale — why conformal after the σ²_v null. |
| `configs/picasso.yaml` | The **production** config (signal = `men_mean_entropy`, mean-matched). Verify every path. |
| `configs/picasso_variance.yaml` | The **ablation** config (signal = `logvol_var`, raw). Verify every path. |
| This guide | Prerequisites + launch steps + output layout. |

The two production configs are independent runs with **separate** `output_dir` and
`logs_dir` — submit them as two launcher invocations (§3.4).

---

## 2. Prerequisites on Picasso (all three are blocking)

The launcher builds the task manifest with its own Python *before* submitting — that
call loads the H5 cohort and imports `crepes`. If either is missing, the launcher
fails immediately at `[1/3] Building task manifest`. Fix all three first.

### 2.1 Sync the merged H5 (carries `uncertainty/per_member_volumes`)

`picasso.yaml` / `picasso_variance.yaml` both point at:

```
/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/MenGrowth.h5
```

The merged file with the `uncertainty/per_member_volumes` group lives locally at:

```
/media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5
```

An older Picasso copy may lack the `uncertainty` group entirely — `ensemble_bma` and
the loader's `mean_m log1p(V_m) == logvol_mean` consistency assertion will fail
without it. rsync the merged file first (run from a `tmux` session):

```bash
rsync -avP --inplace \
  /media/mpascual/MeningD2/MENINGIOMAS/MENGROWTH/050526/h5_format/MenGrowth.h5 \
  tic_163_uma@picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/MenGrowth.h5
```

Then on Picasso, confirm the group is present:

```bash
python -c "import h5py; f=h5py.File('/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/MenGrowth.h5'); print(list(f['uncertainty'].keys()))"
# expect: ['logvol_mean', 'logvol_var', 'men_mean_entropy', 'per_member_volumes', ...]
```

### 2.2 Install `crepes` + `pyarrow` in the Picasso `growth` env

These two dependencies are declared in `pyproject.toml` but are **not** installed by
a plain `conda activate growth` — they were missing in the local env too and had to
be added by hand. On Picasso:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate growth
pip install "crepes>=0.7.0" "pyarrow>=14.0.0"
python -c "import crepes, pyarrow, numpy; print('crepes', crepes.__version__, '| pyarrow', pyarrow.__version__, '| numpy', numpy.__version__)"
# numpy must stay 1.26.x — crepes/pyarrow do not force a numpy 2.0 upgrade.
```

`crepes` backs the `cqr_norm` / `cqr_proper` calibration layers; `pyarrow` backs the
`results_table.parquet` and `per_patient_table.parquet` writers (CSV fallback exists
but parquet is the documented output).

### 2.3 Sync the repository

Push the local branch and pull on Picasso (the repo at
`/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model`), **or** rsync
the working tree. The changes that must reach Picasso (none touch `src/external/`):

```
experiments/stage1_volumetric/conformal_calibration/
  modules/runner.py        # writes per_patient_metrics.json
  modules/aggregator.py    # collect_per_patient + write_per_patient_table
  modules/figures.py       # figure_per_patient_intervals + per-patient width_vs_sigmav
  run.py                   # wires per-patient aggregation into --analyze
  configs/*.yaml           # per_patient_intervals figure registered; variance logs_dir fixed
  tests/test_config_and_manifest.py
```

---

## 3. Launch

All commands are run **from the repo root** on the Picasso login node
(`/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model`). The launcher
auto-activates the `growth` env (read from `slurm.conda_env`).

### 3.1 Dry-run (prints every `sbatch` line, submits nothing)

```bash
bash experiments/stage1_volumetric/conformal_calibration/slurm/launcher.sh \
  experiments/stage1_volumetric/conformal_calibration/configs/picasso.yaml --dry-run
```

Confirm: `N tasks: 60 (array 0-59%40)` — 3 base models × 20 seeds. Constraint shows
`cpu`, partition `gputhin`.

### 3.2 Submit the production run (entropy signal)

```bash
bash experiments/stage1_volumetric/conformal_calibration/slurm/launcher.sh \
  experiments/stage1_volumetric/conformal_calibration/configs/picasso.yaml
```

The launcher submits a 60-task array (`confcal_array`) and a dependent analysis job
(`confcal_analysis`, `--dependency=afterany`) that runs `run.py --analyze` once the
array drains.

### 3.3 Monitor

```bash
squeue -u $USER
tail -f /mnt/home/users/tic_163_uma/mpascual/execs/growth/conformal_calibration/logs/confcal_<ARRAY_ID>_0.out
```

Per-task wall time: LME tasks ~2–3 min, `ensemble_bma` tasks ~25–50 min (×M=20). The
`0-02:00:00` budget in `picasso.yaml` covers the slowest task.

### 3.4 Submit the ablation run (logvol_var signal)

Independent `output_dir` / `logs_dir`, so it can run concurrently:

```bash
bash experiments/stage1_volumetric/conformal_calibration/slurm/launcher.sh \
  experiments/stage1_volumetric/conformal_calibration/configs/picasso_variance.yaml
```

---

## 4. Outputs (what to pull back)

```
{output_dir}/                # picasso.yaml: .../conformal_calibration/entropy
  cohort_meta.json
  manifest.json
  runs/{base_model}/seed_{NNN}/
    conformal_lopo_results.json   # per-patient, per-layer intervals (full fold records)
    marginal_metrics.json         # per-layer IS@95, cov95, width, CRPS, R²_log
    tertile_metrics.json          # the same, stratified by σ²_v tertile
    per_patient_metrics.json      # NEW — long-form per-patient × per-layer rows
  aggregated/
    results_table.parquet         # long-form aggregate metrics
    per_patient_table.parquet     # NEW — every (model, seed, patient, layer) row
    statistics.json               # paired BCa bootstrap + Wilcoxon + Cohen's d + BH-FDR
  figures/
    is_by_model_calibration.png
    coverage_by_model_calibration.png
    tertile_panel.png
    width_vs_sigmav.png           # now a true per-patient scatter (was a tertile proxy)
    per_patient_intervals.png     # NEW — the headline figure
```

### 4.1 The per-patient deliverables (the focus of this preparation)

**`per_patient_table.parquet`** — one row per `(base_model, seed, patient_id, layer)`,
16 columns: `model_name`, `tertile`, `time`, `actual`, `pred_mean`, `pred_var`,
`lower`, `upper`, `width`, `covered`, `interval_score` (per-patient IS@95),
`sigma_v_sq_target`. This is the substrate for any patient-level comparison between
`lme_homo`, `lme_hetero` and `ensemble_bma` — every model, every seed, every held-out
patient, every calibration layer, with its interval and its Winkler score.

**`per_patient_intervals.png`** — the headline figure: a `base_model × layer` grid of
caterpillar panels. In each panel the held-out patients are sorted by σ²_v; every
patient contributes one vertical bar = its prediction interval, coloured by its
per-patient IS@95 (shared log-scaled colour bar); the grey tick is the point
prediction; the observed value is a white dot if the interval covered it, a red ✕ if
it missed. The seed shown is `reporting.per_patient_seed` (0). This is the
"prediction interval for new points per patient + the IS value given them" plot.

Pull the whole tree back with:

```bash
rsync -avP \
  tic_163_uma@picasso:/mnt/home/users/tic_163_uma/mpascual/execs/growth/conformal_calibration/ \
  <local_results_dir>/conformal_calibration/
```

---

## 5. Verification performed before this guide

| Check | Result |
|-------|--------|
| Experiment tests `tests/test_config_and_manifest.py` (incl. 9 new per-patient tests) | **39/39 pass** |
| Core conformal tests `test_conformal.py`, `test_ensemble_lme.py`, `test_conformal_lopo.py` | **42/42 pass** |
| Regression markers `pytest -m "evaluation or phase4"` | **309 passed, 8 skipped, 0 failed** |
| Local smoke `--smoke --force` on `local_smoke.yaml` (real H5, N=24, M=5, 2 seeds) | exit 0; all 6 tasks; `per_patient_metrics.json` ×6; `per_patient_table.parquet` (576 rows); `per_patient_intervals.png` rendered |
| `per_patient_table.parquet` integrity | 576 rows = 6 tasks × 24 patients × 4 layers, 0 nulls, `interval_score` finite (range 0.96–115.7, the upper tail = miss-inflated IS as expected) |

### SLURM review notes

- `launcher.sh`, `worker.sh`, `analysis_worker.sh` mirror the established
  `main_experiment/slurm/` convention (same manifest-driven array + dependent
  analysis pattern).
- `launcher.sh` passes `--constraint=cpu` explicitly on every `sbatch` — the
  documented Picasso lua-plugin workaround. Partition `gputhin` + constraint `cpu`
  matches `main_experiment/configs/picasso.yaml`.
- `sbatch` command-line flags in `launcher.sh` override the `#SBATCH` directives
  inside `worker.sh` / `analysis_worker.sh` (SLURM precedence) — the in-script
  directives are only used if a worker is ever run standalone.
- **Fixed:** `picasso_variance.yaml` `logs_dir` was a copy-paste of `entropy_logs`;
  corrected to `variance_logs` so the two runs do not share a log directory.
- The launcher must be invoked **from the repo root** with a repo-relative config
  path (the worker `cd`s to `REPO_DIR` and reuses `CONFIG_PATH` relative to it).
