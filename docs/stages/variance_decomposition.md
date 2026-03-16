# Variance Decomposition Protocol

## Overview

The variance decomposition is the **central analytical contribution** of the thesis. It formally quantifies how much predictive variance each pipeline component explains, answering the question: *was the complexity worth it?* Rather than presenting Stage 3 as the primary result and hoping it works, we build an evidence ladder where each step must earn its place.

This protocol runs all models under identical LOPO-CV folds, collects per-patient predictions, and computes marginal contributions with statistical significance tests.

---

## Purpose

1. Quantify the marginal $\Delta R^2$ of each model component.
2. Test statistical significance of each improvement via paired permutation tests.
3. Provide bootstrap confidence intervals on all metrics.
4. Identify which patients benefit most from each model upgrade.
5. Produce publication-ready tables and figures for the thesis.

---

## Model Hierarchy

| Model | Input | Effective Params | Notation | Stage |
|---|---|---|---|---|
| $M_0$ | Population mean only | 2 | $\hat{y}^{(0)}$ | Baseline |
| $M_1$ | Volume + time (ScalarGP) | 5 | $\hat{y}^{(1)}$ | Stage 1 |
| $M_2$ | Volume + time (HGP) | 6--8 | $\hat{y}^{(2)}$ | Stage 1 |
| $M_3$ | Volume + time + severity | 3 + N | $\hat{y}^{(3)}$ | Stage 2 |
| $M_4$ | Volume + time + severity + deep features | 3 + N + k | $\hat{y}^{(4)}$ | Stage 3 |

**$M_0$: Population Mean.** $\hat{y}^{(0)}_i = \bar{y}_{\text{train}}$ (constant prediction). 2 effective parameters (mean of training set, implicit variance). Serves as the null model.

**$M_1$: ScalarGP.** Pooled GP with Matern-5/2 + linear mean. 5 hyperparameters. Tests whether time + volume structure explains variance.

**$M_2$: HGP.** LME population mean + per-patient GP conditioning. 6--8 hyperparameters. Tests whether patient-level random effects add signal.

**$M_3$: Severity Model.** Reduced Gompertz with latent severity. 3 population params + N severity values. Tests whether a severity axis captures more than random effects.

**$M_4$: Deep + Severity.** Stage 3 deep features feeding into severity estimation and/or GP with ARD. 3 + N + k params. Tests whether deep features add beyond volume + severity.

---

## Mathematical Framework

### LOPO-CV $R^2$

For model $M_k$, LOPO-CV produces one prediction per held-out patient (under the `last_from_rest` protocol):

$$R^2_k = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}^{(k)}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

where $y_i$ is the actual last-timepoint log-volume and $\hat{y}^{(k)}_i$ is model $M_k$'s prediction, conditioned on all preceding timepoints. $\bar{y}$ is the overall mean of actuals.

**Important:** $R^2$ is computed on the LOPO predictions (out-of-sample), not on the training fit. Negative $R^2$ is possible and means the model is worse than predicting the mean.

### Marginal Contribution

$$\Delta R^2_k = R^2_k - R^2_{k-1}$$

A positive $\Delta R^2_k$ means model $M_k$ explains additional variance beyond $M_{k-1}$.

### Statistical Significance

Use a **paired permutation test** on per-patient squared errors:

$$e^{(k)}_i = (y_i - \hat{y}^{(k)}_i)^2$$

$$H_0: \mathbb{E}[e^{(k-1)}_i - e^{(k)}_i] = 0$$

For each of $B = 10{,}000$ permutations:
1. For each patient $i$, randomly swap $e^{(k-1)}_i$ and $e^{(k)}_i$ with probability 0.5.
2. Compute the difference in mean squared error: $\Delta_b = \overline{e^{(k-1)}}_b - \overline{e^{(k)}}_b$.
3. $p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}[\Delta_b \geq \Delta_{\text{obs}}]$.

**Package:** `scipy.stats.permutation_test` (scipy >= 1.9).

### Bootstrap Confidence Intervals

For each metric, compute BCa bootstrap CIs from the per-patient error vector:

```python
from scipy.stats import bootstrap

def bootstrap_metric(
    per_patient_errors: np.ndarray,   # [N_patients]
    metric_fn: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """BCa bootstrap CI on a metric computed from per-patient errors."""
    result = bootstrap(
        (per_patient_errors,),
        metric_fn,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method='BCa',
        random_state=seed,
    )
    return {
        "point": float(metric_fn(per_patient_errors)),
        "lower": float(result.confidence_interval.low),
        "upper": float(result.confidence_interval.high),
        "se": float(result.standard_error),
    }
```

---

## Input Contract

```python
# Per-model LOPO predictions (all models run on identical folds)
model_predictions: dict[str, dict[str, float]]
# Maps model_name -> {patient_id: predicted_value}

# Actual values
actuals: dict[str, float]
# Maps patient_id -> actual last-timepoint log-volume

# Per-model LOPO squared errors
model_errors: dict[str, np.ndarray]
# Maps model_name -> [N_patients] array of (y_i - y_hat_i)^2

# Stage results for enrichment
stage1_results: LOPOResults  # Best Stage 1 model
stage2_results: dict         # Severity model results
stage3_results: dict         # Deep feature results (if available)
```

## Output Contract

```python
# Variance decomposition results
decomposition: dict = {
    "models": [
        {
            "name": str,                    # "M0_PopMean", "M1_ScalarGP", etc.
            "stage": int,                   # 0, 1, 1, 2, 3
            "n_effective_params": int,
            "r2_log": float,
            "r2_log_ci_lower": float,
            "r2_log_ci_upper": float,
            "delta_r2": float | None,       # None for M0
            "delta_r2_p_value": float | None,
            "mae_log": float,
            "mae_log_ci_lower": float,
            "mae_log_ci_upper": float,
            "rmse_log": float,
            "calibration_95": float | None, # None for M0
            "per_patient_errors": list[float],
        },
        # ... one entry per model
    ],
    "pairwise_tests": [
        {
            "model_a": str,
            "model_b": str,
            "delta_r2": float,
            "p_value": float,
            "significant_at_005": bool,
            "significant_at_001": bool,
        },
        # ... one entry per adjacent pair
    ],
    "patient_analysis": {
        "patients_improved_m1_vs_m0": list[str],
        "patients_improved_m2_vs_m1": list[str],
        "patients_improved_m3_vs_m2": list[str],
        "patients_improved_m4_vs_m3": list[str],
        "patients_degraded_by_complexity": list[str],
    },
    "summary": {
        "best_model": str,
        "best_r2": float,
        "largest_delta_r2": float,
        "largest_delta_model": str,
        "stage_justification": {
            "stage2_justified": bool,       # p < 0.05 for M3 vs M2
            "stage3_justified": bool,       # p < 0.05 for M4 vs M3
        },
    },
}

# Output files
# experiments/variance_decomposition/results/
#   decomposition_results.json
#   decomposition_table.tex           # LaTeX table for thesis
#   delta_r2_bar_chart.pdf
#   per_patient_error_scatter.pdf
#   model_comparison_table.pdf
```

---

## Output Space

**Primary:** Log-volume space ($y = \log(V + 1)$). This is clinically interpretable: a prediction error of 0.1 in log-space corresponds to ~10% volume error.

**Secondary:** Quantile space (for Stage 2 severity model). Report both, with log-volume as the space for cross-model comparison.

**Conversion for Stage 2:** The severity model predicts in quantile space. To compare with other models, inverse-transform predictions to log-volume using the empirical CDF (computed on training data per LOPO fold).

---

## Reporting Format

### Summary Table (for thesis)

```
| Model           | R^2_log        | Delta R^2 | p-value | MAE_log        | Cal_95 |
|-----------------|----------------|-----------|---------|----------------|--------|
| M0: PopMean     | ...  [CI]      | --        | --      | ...  [CI]      | --     |
| M1: ScalarGP    | ...  [CI]      | ...       | ...     | ...  [CI]      | ...    |
| M2: HGP         | ...  [CI]      | ...       | ...     | ...  [CI]      | ...    |
| M3: Severity    | ...  [CI]      | ...       | ...     | ...  [CI]      | ...    |
| M4: Deep+Sev    | ...  [CI]      | ...       | ...     | ...  [CI]      | ...    |
```

### Figures

1. **Delta R^2 bar chart with bootstrap 95% CIs** -- Shows marginal contribution of each model. Error bars from bootstrap. Horizontal dashed line at Delta R^2 = 0. Stars for significance (* p < 0.05, ** p < 0.01).

2. **Per-patient error scatter** -- X-axis: patient index (sorted by n_timepoints). Y-axis: absolute error |y - y_hat|. One series per model, connected by lines. Highlights patients that improve/degrade with complexity.

3. **Cumulative R^2 stacked bar** -- Shows how R^2 accumulates from M0 to M4. Each bar segment = Delta R^2_k.

---

## Code Requirements

### New Files

1. **`src/growth/evaluation/variance_decomposition.py`** -- Core decomposition logic.

   ```python
   @dataclass
   class ModelEntry:
       """Results for one model in the decomposition."""
       name: str
       stage: int
       n_effective_params: int
       predictions: dict[str, float]      # patient_id -> predicted value
       per_patient_errors: np.ndarray      # [N_patients] squared errors

   class VarianceDecomposition:
       """Computes variance decomposition across a model hierarchy.

       Args:
           actuals: Dict mapping patient_id to actual value.
           n_bootstrap: Number of bootstrap resamples for CIs.
           n_permutations: Number of permutations for significance tests.
           seed: Random seed.
       """

       def __init__(
           self,
           actuals: dict[str, float],
           n_bootstrap: int = 2000,
           n_permutations: int = 10000,
           seed: int = 42,
       ) -> None: ...

       def add_model(
           self,
           name: str,
           stage: int,
           n_params: int,
           predictions: dict[str, float],
       ) -> None:
           """Add a model's LOPO predictions to the decomposition."""

       def compute(self) -> dict:
           """Run full decomposition: R^2, Delta R^2, CIs, permutation tests.

           Returns dict matching output contract.
           """

       def _compute_r2(self, predictions: dict[str, float]) -> float:
           """R^2 from per-patient predictions vs actuals."""

       def _paired_permutation_test(
           self,
           errors_a: np.ndarray,
           errors_b: np.ndarray,
       ) -> float:
           """Two-sided paired permutation test. Returns p-value."""

       def _bootstrap_ci(
           self,
           errors: np.ndarray,
           metric_fn: Callable,
       ) -> dict:
           """BCa bootstrap CI. Returns {point, lower, upper, se}."""

       def generate_table_latex(self, results: dict) -> str:
           """Generate LaTeX table for thesis."""

       def generate_figures(
           self, results: dict, output_dir: str
       ) -> list[str]:
           """Generate all figures. Returns list of file paths."""
   ```

2. **`experiments/variance_decomposition/run_decomposition.py`** -- Orchestrator.

   ```python
   def main():
       """Run variance decomposition across all stages.

       Steps:
         1. Load all patient trajectories
         2. Run M0 (population mean) LOPO predictions
         3. Run M1 (ScalarGP) LOPO-CV -- or load from Stage 1 results
         4. Run M2 (HGP) LOPO-CV -- or load from Stage 1 results
         5. Run M3 (Severity) LOPO-CV -- or load from Stage 2 results
         6. Run M4 (Deep+Severity) LOPO-CV -- or load from Stage 3 results
         7. Feed all predictions to VarianceDecomposition
         8. Compute decomposition
         9. Generate table and figures
         10. Save results
       """
   ```

3. **`experiments/variance_decomposition/config.yaml`** -- Configuration.

### Modifications to Existing Files

- `src/growth/evaluation/__init__.py`: Export `VarianceDecomposition`.

---

## Implementation Detail

### M0: Population Mean Baseline

```python
def compute_m0_predictions(
    patients: list[PatientTrajectory],
) -> dict[str, float]:
    """Predict last timepoint as the training-set mean (LOPO).

    For each held-out patient, the prediction is the mean of all other
    patients' last timepoints.
    """
    predictions = {}
    for i, held_out in enumerate(patients):
        train = [p for j, p in enumerate(patients) if j != i]
        train_means = [p.observations[-1, 0] for p in train]
        predictions[held_out.patient_id] = float(np.mean(train_means))
    return predictions
```

### Collecting Predictions from Existing Stages

Each stage's `LOPOResults` contains per-fold predictions. Extract them:

```python
def extract_lopo_predictions(
    lopo_results: LOPOResults,
    protocol: str = "last_from_rest",
) -> dict[str, float]:
    """Extract per-patient predictions from LOPOResults.

    Returns dict mapping patient_id to predicted value.
    """
    predictions = {}
    for fold in lopo_results.fold_results:
        if protocol in fold.predictions:
            for pred_dict in fold.predictions[protocol]:
                predictions[fold.patient_id] = pred_dict["pred_mean"]
    return predictions
```

### Ensuring Identical Folds

All models MUST use the same LOPO folds (same patient exclusion list, same ordering). This is guaranteed by:
1. Using the same `patients` list for all models.
2. Using the same `exclude` list from config.
3. Using `LOPOEvaluator` consistently (iterates over patients in order).

---

## Configuration

```yaml
# experiments/variance_decomposition/config.yaml
experiment:
  name: variance_decomposition
  seed: 42

paths:
  mengrowth_h5: /path/to/MenGrowth.h5
  output_dir: experiments/variance_decomposition/results
  stage1_results: experiments/segment_based_approach/results
  stage2_results: experiments/severity_model/results
  stage3_results: experiments/deep_features/results

models:
  m0_popmean: true
  m1_scalar_gp: true
  m2_hgp: true
  m3_severity: true
  m4_deep_severity: true     # Only if Stage 3 is complete

# Reuse results from stages (avoid re-running)
reuse_stage_results: true     # Load from stage result dirs if available

statistics:
  n_bootstrap: 2000
  bootstrap_method: bca       # bca | percentile
  n_permutations: 10000
  significance_alpha: 0.05
  seed: 42

output_space:
  primary: log_volume         # log(V + 1)
  secondary: quantile         # For Stage 2 comparison

figures:
  format: [pdf, png]
  dpi: 300
  font_size: 12
  font_family: serif

patients:
  exclude: [MenGrowth-0028]
  min_timepoints: 2

# Protocol for extracting predictions
prediction_protocol: last_from_rest
```

---

## Verification Tests

All decomposition tests are **DIAGNOSTIC** -- they report results but do not block.

```
VD-T1: All models produce predictions for all patients [BLOCKING]
  - Assert each model has N predictions (one per patient)
  - Assert all predictions are finite (no NaN/Inf)
  - Assert patient IDs match across all models
  Recovery: Re-run failed model; check for fold failures

VD-T2: R^2 ordering is weakly monotonic [DIAGNOSTIC]
  - Check R^2_0 <= R^2_1 <= R^2_2 (expected but not guaranteed for LOPO)
  - If violated, report which model degrades and by how much
  - Violation is scientifically informative (overfitting with complexity)
  Note: DIAGNOSTIC -- non-monotonicity is a valid finding

VD-T3: Permutation test implementation is correct [BLOCKING]
  - Test with identical error vectors: assert p = 1.0
  - Test with clearly different errors (one vector shifted by 10):
    assert p < 0.01
  - Assert p in [0, 1] for all model pairs
  Recovery: Check permutation logic; verify sign convention

VD-T4: Bootstrap CIs contain point estimate [BLOCKING]
  - For each model, assert CI_lower <= R^2_point <= CI_upper
  - Assert CI width > 0 (non-degenerate)
  Recovery: Increase n_bootstrap; check for degenerate error vectors

VD-T5: LaTeX table and figures generated [DIAGNOSTIC]
  - Assert .tex file is valid LaTeX (no syntax errors)
  - Assert .pdf figures exist and have size > 0
  - Visual inspection is manual
  Note: DIAGNOSTIC -- formatting issues do not affect results

VD-T6: Cross-stage consistency [DIAGNOSTIC]
  - If loading from stage results, verify predictions match re-computed ones
  - Spot-check 3 patients: loaded prediction == re-computed prediction
  Note: DIAGNOSTIC -- detects stale cached results
```

---

## Interpretation Guide

### If $\Delta R^2_1 > 0$ and $p < 0.05$
The ScalarGP captures temporal structure beyond the population mean. Expected to hold.

### If $\Delta R^2_2 > 0$ but $p > 0.05$
HGP patient-level conditioning helps some patients but the improvement is not significant. Given N=31, this is a power issue. Report the effect size.

### If $\Delta R^2_3 > 0$ and $p < 0.05$
The severity model justifies its added complexity. The latent severity axis captures meaningful inter-patient variability. This validates the advisor's proposal.

### If $\Delta R^2_3 \leq 0$
The severity model does NOT outperform HGP. Possible explanations: (a) N=31 is too small for 34 parameters, (b) the quantile transform loses too much information, (c) meningioma growth variability is captured by random effects alone.

### If $\Delta R^2_4 > 0$ and $p < 0.05$
Deep features capture growth-relevant information beyond volume. This justifies the full BrainSegFounder pipeline. Examine ARD lengthscales to understand WHAT the features capture.

### If $\Delta R^2_4 \leq 0$
Deep features add nothing beyond volume for growth prediction at N=31. This is a valid negative result and a scientifically important finding. It means meningioma growth (at this sample size) is fully explained by volume trajectories.

---

## References

- Efron, B. & Tibshirani, R. "Improvements on Cross-Validation: The .632+ Bootstrap Method," *JASA*, 1997.
- Good, P. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, Springer, 2005.
- Boulesteix, A.-L. et al. "Overview of Random Forest Methodology and Practical Guidance with Emphasis on Computational Biology and Bioinformatics," *WIREs Data Mining and Knowledge Discovery*, 2012. (variance decomposition in predictive modeling)
- Hastie, T. et al. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009. (bias-variance tradeoff, model selection)
