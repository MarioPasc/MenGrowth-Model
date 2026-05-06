# Synthetic σ²_v Stress Test (Stage 1 UQ)

Implements `docs/UQ_SYNTHETIC_VARIANCE_STRESSTEST.md`. Tests whether the
hetero/homo conditional-calibration gap reported in
`docs/UQ_HETERO_CALIBRATION_ANSWER.md` is a *causal* property of the
σ²_v dispersion or an artefact of correlated nuisance variables.

## Run

```bash
cd /home/mpascual/research/code/MenGrowth-Model

# Smoke test (3 levels × 2 seeds, ≈ 90 s)
PYTHONPATH=src ~/.conda/envs/growth/bin/python \
  -m experiments.stage1_volumetric.synthetic_uq.run_synthetic_uq --smoke

# Full sweep (16 levels × 10 seeds, ≈ 40 min)
PYTHONPATH=src ~/.conda/envs/growth/bin/python \
  -m experiments.stage1_volumetric.synthetic_uq.run_synthetic_uq --n-seeds 10

# Aggregate + figures
PYTHONPATH=src ~/.conda/envs/growth/bin/python \
  -m experiments.stage1_volumetric.synthetic_uq.aggregate
```

## What the sweep does

For every `(profile, level, seed)`:

1. Reuse the cohort loaded from `MenGrowth.h5` (56 patients, 173 scans
   after `MenGrowth-0028` exclusion).
2. Sample a synthetic σ²_v vector of length 173 from the profile.
3. Overwrite each `traj.observation_variance` and run LOPO
   `last_from_rest` for **LMEHetero**.
4. Reuse a once-cached **LME (homo)** baseline for the same
   trajectories — homo LME ignores `observation_variance`, so the same
   predictions apply to every (profile, level, seed).
5. Persist per-fold predictions, marginal metrics, and per-tertile
   conditional metrics. Tertiles are defined by the **empirical**
   σ²_{v,*} so all profiles share the same patient strata (otherwise the
   tertile boundaries would shift with the injected vector and confound
   the comparison).

## Profiles

| Profile | What it tests | Levels |
|---|---|---|
| **A — constant** | Parity check: σ²_v = c for all scans (degenerate dispersion). LMEHetero should match LME marginally at any c. | c ∈ {1e-3, 1e-2, 1e-1, 1.0} |
| **B — matched empirical** | Reproduces the empirical bimodal distribution (mean ≈ 0.42, p_high ≈ 6.1%). | 1 |
| **C — bimodal p sweep** | High-tail fraction p ∈ {0, 0.05, 0.10, 0.20, 0.40} with cohort mean fixed. Causal test: hetero rescues high-tertile coverage as p grows. | 5 |
| **D — log-normal τ sweep** | Continuous dispersion sweep (μ = log(0.42) − τ²/2). Smooth dispersion without bimodality. | τ ∈ {0, 0.5, 1.0, 1.5, 2.0} |
| **E — empirical** | Pass-through of the empirical σ²_v. Sanity baseline; should match `conditional_calibration_last_from_rest.json`. | 1 |

## Outputs

```
synthetic_uq/
├── cohort_meta.json              # patient list, n_timepoints_per_patient, tertile cuts
├── lme_baseline.json             # cached LME LOPO results (used for every profile/seed)
├── runs/
│   └── {profile}_{level}/seed{NNN}/
│       ├── marginal.json          # rows for LME and LMEHetero
│       ├── sigma_v_sq_injected.npy
│       └── DONE
├── summary_rows.json             # all rows across the sweep
├── aggregated/
│   ├── marginal_table.csv
│   ├── marginal_summary.csv
│   ├── conditional_table.csv
│   ├── conditional_summary.csv
│   └── paired_high_tertile.csv   # paired bootstrap of Δ on high tertile
└── figures/
    ├── fig_A_constant.{pdf,png}
    ├── fig_C_p_sweep_high.{pdf,png}
    └── fig_D_tau_sweep_high.{pdf,png}
```

## Idempotency

Each run writes a `DONE` marker. Re-running the sweep skips finished
`(profile, level, seed)` directories. To re-run a subset, delete
`runs/{profile}_{level}/seed*/DONE`.
