# Stage 1 -- Segmentation-Based Volumetric Baseline

## Overview

Stage 1 establishes the **strong empirical baseline** against which all subsequent stages are compared. It answers the question: *how far can scalar volume trajectories take us?* This is not a throwaway ablation -- it is a first-class model that may well be the final answer if higher-complexity stages fail to justify their additional parameters.

The pipeline extracts whole-tumor (WT) volume from each MRI via BrainSegFounder segmentation, transforms to log-space, and fits three growth models of increasing sophistication under LOPO-CV.

---

## Pipeline

```
MenGrowth MRI [4, D, H, W]
  --> BrainSegFounder (frozen / adapted decoder / LoRA)
  --> Sliding-window inference (128^3, overlap=0.5, Gaussian weighting)
  --> Argmax --> Binary WT mask (labels 1 | 2 | 3)
  --> Volume = sum(mask > 0) * voxel_volume_mm^3
  --> y = log(V + 1)                          [log1p transform]
  --> Per-patient trajectory: {(t_k, y_k)}_{k=1}^{n_i}
```

Volume extraction uses `compute_volumes()` and `compute_log_volumes()` from `src/growth/data/semantic_features.py`. Voxel volume is 1.0 mm^3 (isotropic after BraTS preprocessing).

---

## Input Contract

```python
# MenGrowth H5 file (unified v2.0 schema)
h5_path: str  # Path to MenGrowth.h5

# Segmentation model checkpoints (configured in config.yaml)
seg_models: list[SegModelConfig]  # From experiments/segment_based_approach/segment.py

# Per-patient trajectories (built from H5 longitudinal structure)
trajectories: list[PatientTrajectory]
# Each has:
#   patient_id: str
#   times: np.ndarray       [n_i], ordinal or days-from-baseline
#   observations: np.ndarray [n_i, 1], y = log(V + 1)
#   covariates: dict | None  e.g. {"age": 65, "sex": 0}
```

## Output Contract

```python
# Per growth model, per segmentation source
lopo_results: dict[str, LOPOResults]
# Key format: "{GrowthModel}_{source}", e.g. "ScalarGP_manual", "LME_bsf_lora_r8"

# Aggregate metrics (per protocol, per model)
metrics: dict = {
    "last_from_rest/r2_log": float,        # Primary metric
    "last_from_rest/mae_log": float,
    "last_from_rest/rmse_log": float,
    "last_from_rest/calibration_95": float,
    "last_from_rest/r2_original": float,
    "last_from_rest/mae_original": float,
    "all_from_first/r2_log": float,
    "all_from_first/mae_log": float,
    "per_patient_r_mean": float,
    "per_patient_r_std": float,
}

# Bootstrap CIs on primary metric (NEW)
bootstrap_ci: dict = {
    "r2_log_point": float,
    "r2_log_lower_95": float,
    "r2_log_upper_95": float,
    "method": "bootstrap_632plus",
    "n_bootstrap": 2000,
}

# Output files
# experiments/segment_based_approach/results/
#   segmentation/comparison.json
#   growth_prediction/{source}/lopo_results_{model}.json
#   growth_prediction/{source}/model_comparison.json
#   growth_prediction/cross_source_comparison.json
#   figures/segmentation/*.png
#   figures/growth_prediction/{source}/*.png
```

---

## Three Growth Models

### Model A: ScalarGP

**File:** `src/growth/models/growth/scalar_gp.py`
**Library:** GPy
**Interface:** `GrowthModel` (from `src/growth/models/growth/base.py`)

All patients pooled into a single GP. At prediction time, the GP is conditioned on the held-out patient's observations.

$$y_i(t) \sim \mathcal{GP}\bigl(\beta_0 + \beta_1 t,\; k_{\text{Mat52}}(t, t') + \sigma^2_n \delta(t, t')\bigr)$$

$$k_{\text{Mat52}}(r) = \sigma^2_f \left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5\,r^2}{3\,\ell^2}\right) \exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right)$$

**Parameters:** 5 ($\beta_0, \beta_1, \sigma^2_f, \ell, \sigma^2_n$). Within budget at N=31.

**Configuration:**
```yaml
gp:
  kernel: matern52
  mean_function: linear
  n_restarts: 5
  max_iter: 1000
  lengthscale_bounds: [0.1, 50.0]
  signal_var_bounds: [0.001, 10.0]
  noise_var_bounds: [0.000001, 5.0]
```

### Model B: LME

**File:** `src/growth/models/growth/lme_model.py`
**Library:** `statsmodels` (REML via `MixedLM`)

Per-patient random intercept and slope with population-level fixed effects:

$$y_{ij} = (\beta_0 + u_{0i}) + (\beta_1 + u_{1i})\,t_{ij} + \varepsilon_{ij}$$

where $(u_{0i}, u_{1i})^T \sim \mathcal{N}(0, \Omega)$, $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$.

**Parameters:** 6 (2 fixed + 3 variance components + 1 residual). BLUP prediction with automatic shrinkage.

**Fallback chain:** random intercept+slope --> random intercept only --> OLS (implemented in `lme_model.py`).

### Model C: HGP (Hierarchical GP)

**File:** `src/growth/models/growth/hgp_model.py`
**Library:** GPy

Population linear mean from LME (D18) plus Matern-5/2 on residuals:

$$y_i(t) = \hat{\beta}_0 + \hat{\beta}_1 t + f_i(t) + \varepsilon_i(t)$$

where $f_i(t) \sim \mathcal{GP}(0, k_{\text{Mat52}})$ captures individual deviations. Kernel hyperparameters shared across patients via empirical Bayes.

**Parameters:** LME fixed effects (from Model B, frozen) + 3 kernel hyperparameters ($\sigma^2_f, \ell, \sigma^2_n$).

---

## Covariates (When Available)

When timestamps, age, and sex metadata arrive, incorporate as fixed effects in the GP mean function:

$$m(t; \mathbf{x}_i) = \beta_0 + \beta_1 t + \beta_2 \cdot \text{age}_i + \beta_3 \cdot \text{sex}_i$$

**Implementation:** Already supported via `covariate_utils.py`. Enable in config:
```yaml
covariates:
  enabled: true
  features: [age, sex]
  missing_strategy: skip
```

**Parameter impact:** +2 mean-function parameters. Tight at N=31, acceptable at N=58.

---

## Improvements to Implement

### 1. Gompertz Mean Function for HGP

Literature (Vaghi et al. 2020; Engelhardt et al. 2023) establishes Gompertz as the best parametric model for meningioma growth. Replace the linear GP mean with:

$$m(t) = V_{\max} \exp\!\bigl(-\exp(-\alpha(t - t_{\text{mid}}))\bigr)$$

**Implementation strategy (zero additional GP parameters):**
1. Fit Gompertz parametrically to all training patients via `scipy.optimize.curve_fit` on pooled $(t, y)$ data.
2. Compute residuals: $r_{ij} = y_{ij} - m_{\text{Gompertz}}(t_{ij})$.
3. Fit the GP on residuals with zero mean (as HGP currently does with LME residuals).

**Where:** Add `gompertz` option to `hgp_model.py` mean function handling. Add helper `_fit_gompertz_mean()` that returns the fitted parametric curve.

**Ablation:** Compare `hgp.mean_function: linear` vs `hgp.mean_function: gompertz` under LOPO-CV.

### 2. Bootstrap CIs on LOPO-CV Metrics

Current `LOPOEvaluator` reports point estimates only. Add .632+ bootstrap (Efron & Tibshirani, 1997).

**Implementation:**
```python
def bootstrap_lopo_metric(
    per_patient_errors: np.ndarray,  # [N_patients]
    metric_fn: Callable,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap 95% CI on a LOPO-CV metric.

    Uses scipy.stats.bootstrap with BCa method.

    Returns:
        {"point": float, "lower_95": float, "upper_95": float, "se": float}
    """
```

**Where:** `src/growth/evaluation/lopo_evaluator.py` or new file `src/growth/evaluation/bootstrap_ci.py`.

**Package:** `scipy.stats.bootstrap` (scipy >= 1.9).

### 3. Scanner Effect Testing

Before ComBat harmonization, test whether scanner effects are statistically significant for volumetric features.

**Implementation:**
1. Compute log-volume residuals after detrending for time (subtract per-patient linear fit).
2. Kruskal-Wallis test across scanner groups on residuals.
3. If $p > 0.05$, ComBat is unnecessary and may introduce artifacts.

**Package:** `scipy.stats.kruskal`.

**Where:** Add to `experiments/segment_based_approach/run_baseline.py` as a pre-analysis step, or new utility in `src/growth/evaluation/`.

---

## Segmentation Models

Configured in `experiments/segment_based_approach/config.yaml`:

| Model Name | Description | Checkpoint |
|---|---|---|
| `brainsegfounder` | Frozen original BSF | `finetuned_model_fold_0.pt` |
| `bsf_adapted_decoder_men_domain` | Decoder fine-tuned on BraTS-MEN | `baseline/best_model.pt` |
| `bsf_lora_r8_adapted_men_domain` | LoRA r=8 + adapted decoder | `men_r8/best_model.pt` |

Additionally, **manual segmentation** from the MenGrowth H5 serves as the ground-truth volume source.

---

## Existing Code

| File | Purpose |
|---|---|
| `experiments/segment_based_approach/run_baseline.py` | Main orchestrator |
| `experiments/segment_based_approach/segment.py` | `SegmentationVolumeExtractor`, `ScanVolumes`, multi-model support |
| `experiments/segment_based_approach/config.yaml` | Configuration |
| `src/growth/models/growth/scalar_gp.py` | ScalarGP (Model A) |
| `src/growth/models/growth/lme_model.py` | LME (Model B) |
| `src/growth/models/growth/hgp_model.py` | HGP (Model C) |
| `src/growth/models/growth/base.py` | `PatientTrajectory`, `GrowthModel`, `FitResult`, `PredictionResult` |
| `src/growth/models/growth/covariate_utils.py` | Covariate collection and handling |
| `src/growth/evaluation/lopo_evaluator.py` | `LOPOEvaluator`, `LOPOResults`, `LOPOFoldResult` |
| `src/growth/data/semantic_features.py` | `compute_volumes()`, `compute_log_volumes()` |

---

## Configuration

```yaml
# experiments/segment_based_approach/config.yaml
experiment:
  name: stage1_volumetric_baseline
  seed: 42

paths:
  mengrowth_h5: /path/to/MenGrowth.h5
  output_dir: experiments/segment_based_approach/results

segmentation:
  sw_roi_size: [128, 128, 128]
  sw_overlap: 0.5
  sw_mode: gaussian
  wt_threshold: 0.5
  use_manual_segmentation: true
  models_to_use:
    - model_name: brainsegfounder
      type: BrainSegFounder
      checkpoints: /path/to/finetuned_model_fold_0.pt
      enabled: true
    - model_name: bsf_lora_r8_adapted_men_domain
      type: BrainSegFounder
      checkpoints: /path/to/men_r8/best_model.pt
      lora_alpha: 16
      lora_rank: 8
      enabled: true

volume:
  transform: log1p

prediction:
  target: absolute  # absolute | delta_v

covariates:
  enabled: false
  features: [age, sex]
  missing_strategy: skip

gp:
  kernel: matern52
  mean_function: linear  # linear | gompertz (NEW)
  n_restarts: 5
  max_iter: 1000
  lengthscale_bounds: [0.1, 50.0]
  signal_var_bounds: [0.001, 10.0]
  noise_var_bounds: [0.000001, 5.0]

lme:
  method: reml

models:
  scalar_gp: true
  lme: true
  hgp: true

time:
  variable: ordinal  # ordinal | days_from_baseline

patients:
  exclude: [MenGrowth-0028]
  min_timepoints: 2

bootstrap:
  enabled: true
  n_samples: 2000
  method: bca        # bca | percentile
  seed: 42

scanner_test:
  enabled: true      # Run Kruskal-Wallis before ComBat
  alpha: 0.05
```

---

## Verification Tests

```
S1-T1: ScalarGP LOPO-CV completes without NaN [BLOCKING]
  - Run LOPO-CV with ScalarGP on manual volumes
  - Assert all folds produce finite predictions (no NaN/Inf)
  - Assert all folds converge (log-marginal-likelihood is finite)
  Recovery: Check kernel bounds; increase noise variance lower bound to 1e-4

S1-T2: LME R^2_log > 0 [BLOCKING]
  - LME captures temporal trend beyond population mean
  - Assert R^2_log > 0 under last_from_rest protocol
  Recovery: Check data ordering; verify time variable is monotonic per patient

S1-T3: HGP R^2_log >= ScalarGP R^2_log [DIAGNOSTIC]
  - Hierarchical structure should not degrade prediction
  - If violated, investigate whether LME mean function is poorly estimated
  Note: DIAGNOSTIC -- violation is informative but does not block

S1-T4: Calibration_95 in [0.85, 1.0] [DIAGNOSTIC]
  - Prediction intervals are well-calibrated (conservative is acceptable)
  - Report actual coverage for all three models
  Note: DIAGNOSTIC -- poor calibration is logged, not blocking

S1-T5: Bootstrap 95% CI on R^2_log excludes 0 for best model [BLOCKING]
  - Run B=2000 bootstrap resamples of per-patient LOPO errors
  - Assert lower bound of 95% CI > 0 for at least one model
  Recovery: If CI includes 0, sample size is insufficient for any model to
    demonstrate significance. Report this honestly.

S1-T6: Per-patient error distribution is reported [BLOCKING]
  - Save per-patient prediction errors (not just aggregates)
  - Report mean, std, min, max, quartiles of absolute error
  - Generate per-patient error scatter plot
  Recovery: Trivial -- just ensure save logic runs

S1-T7: Gompertz mean function tested as ablation [DIAGNOSTIC]
  - Compare HGP with linear mean vs HGP with Gompertz mean
  - Report Delta R^2 and paired permutation test p-value
  - If Gompertz fails to converge for >20% of folds, report failure mode
  Note: DIAGNOSTIC -- Gompertz may not help with ordinal time
```

---

## References

- Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning*, MIT Press, 2006.
- Laird, N. M. & Ware, J. H. "Random-Effects Models for Longitudinal Data," *Biometrics*, 1982.
- Schulam, P. & Saria, S. "Individualizing Predictions of Disease Trajectories," NeurIPS, 2015.
- Vaghi, C. et al. "Population modeling of tumor growth curves and the reduced Gompertz model improve prediction of the age of experimental tumors," *PLOS Computational Biology*, 2020.
- Engelhardt, S. et al. "Meningioma growth dynamics assessed by volumetric analysis," *Journal of Neuro-Oncology*, 2023.
- Efron, B. & Tibshirani, R. "Improvements on Cross-Validation: The .632+ Bootstrap Method," *JASA*, 1997.
