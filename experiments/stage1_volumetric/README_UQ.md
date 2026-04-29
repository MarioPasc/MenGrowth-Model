# Stage 1 — Uncertainty-Propagated Volume Prediction

Heteroscedastic growth models that propagate per-scan segmentation uncertainty (from the LoRA-ensemble, r=32, M=20) through to growth prediction. Each observation carries a known measurement-error variance σ²_v from the ensemble dispersion, which enters the likelihood as additive noise: R_i = σ²_n I + diag(σ²_v). The homoscedastic baselines are run in parallel for paired comparison.

Full mathematical derivation: `docs/growth-related/uncertainty_propagation_growth_prediction/SPEC_uncertainty_propagated_volume_prediction.md`

## Usage

```bash
# Ordinal time (default)
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml

# Real time (days from baseline)
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml --real-time

# Robust estimator (median/MAD instead of mean/std)
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml --estimator median_mad
```

## Note

The homoscedastic baseline is in `run_stage1.py` / `config.yaml` (unchanged).
