# Stage 2 -- Latent Severity Model

## Overview

Stage 2 introduces the advisor's proposal: a single latent variable $s_i \in [0, 1]$ ("severity") that governs each patient's growth trajectory. Mathematically, this is a **nonlinear mixed-effects model (NLME)** with a single subject-specific parameter. The key hypothesis: inter-patient variability in growth can be captured by a one-dimensional latent axis, enabling individualized prediction from baseline features alone.

Stage 2 is justified **only if** it demonstrably outperforms Stage 1 under LOPO-CV. At N=31, this requires careful parameter budgeting: the population model has 3 parameters, with N additional severity values estimated jointly.

---

## Objective

Test whether a latent severity parameterization outperforms pure volumetric models (Stage 1) for growth prediction. Specifically:
1. Does a severity-indexed growth function explain inter-patient variance better than random effects (LME)?
2. Can severity be predicted from baseline features, enabling test-time individualization?
3. Does the severity ordering correlate with clinical proxies (baseline volume, growth rate)?

---

## Mathematical Formalization

### Core Model

$$q_{ij} = g(s_i, t_{ij};\, \boldsymbol{\theta}) + \varepsilon_{ij}, \quad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$$

where:
- $q_{ij} \in [0, 1]$: growth quantile for patient $i$ at observation $j$
- $s_i \in [0, 1]$: latent severity (constant per patient, estimated from data)
- $t_{ij} \in [0, 1]$: normalized time (quantile of elapsed time from baseline)
- $\boldsymbol{\theta}$: shared population parameters
- $g$ satisfies: (i) $g(s, 0) = 0\ \forall s$, (ii) $\partial g / \partial t \geq 0$, (iii) $\partial g / \partial s \geq 0$

### Connection to Item Response Theory (IRT)

The model has a precise structural analogy to IRT:
- $s_i$ = latent ability (severity)
- $t_{ij}$ = item difficulty (time)
- $q_{ij}$ = response probability (growth quantile)

This analogy provides theoretical grounding: IRT models are well-studied for estimating latent traits from sparse ordinal data.

---

## Quantile Transform Protocol

### Time Quantile $t_{ij}$

Let $\Delta t_{ij}$ be elapsed time from baseline for patient $i$ at observation $j$ (ordinal indices if no timestamps; days otherwise).

$$t_{ij} = \frac{\text{rank}(\Delta t_{ij})}{N_{\text{total}} + 1}$$

where $N_{\text{total}}$ is the total number of observations across all patients. Uses `scipy.stats.rankdata(method='average')`.

**Baseline convention:** $t_{i0} = 0$ exactly (baseline observation excluded from ranking, or defined as rank 0).

### Growth Quantile $q_{ij}$

$$q_{ij} = \hat{F}_{\Delta V}\bigl(\Delta V_{ij}\bigr)$$

where $\Delta V_{ij} = \log(V_{ij} + 1) - \log(V_{i0} + 1)$ (log-ratio growth from baseline). The empirical CDF $\hat{F}$ is computed across all non-baseline observations.

**Baseline convention:** $q_{i0} = 0$ exactly (zero growth at baseline, matching boundary condition $g(s, 0) = 0$).

**Warning:** The quantile transform is rank-based and destroys magnitude information. A patient growing by 1 cm^3 and another by 100 cm^3 may be adjacent in quantile space. This is intentional -- the severity model operates on growth *ordering*, not absolute magnitudes.

### Implementation

```python
def compute_quantiles(
    patient_trajectories: list[PatientTrajectory],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform volume trajectories to (t_quantile, q_growth) space.

    Args:
        patient_trajectories: Trajectories with observations = log(V+1).

    Returns:
        (t_quantiles, q_growths): Arrays [N_total_non_baseline] in [0, 1].
    """
```

**File:** `src/growth/models/growth/quantile_transform.py` (NEW)

---

## Growth Function Options

### Option 1: Weighted Sigmoid (Simple, 3 params)

$$g(s, t; w_1, w_2, b) = t \cdot \sigma(w_1 s + w_2 t + b)$$

where $\sigma$ is the logistic sigmoid and $w_1, w_2 > 0$ (constrained positive). Multiplication by $t$ ensures $g(s, 0) = 0$ exactly.

**Monotonicity:** $\partial g / \partial s = t \cdot \sigma'(\cdot) \cdot w_1 \geq 0$ (exact). $\partial g / \partial t$ is approximately monotone when $w_2 > 0$.

**Parameters:** 3 population ($w_1, w_2, b$) + N severity values.

### Option 2: CMNN (Expressive, ~50 params -- NOT RECOMMENDED at N=31)

Constrained Monotonic Neural Network using `MonoDense` layers from Runje & Shankaranarayana (2023). Guaranteed monotonicity via constrained weights.

**Parameters:** ~50. Catastrophically overparameterized at N=31. Include only as ablation at N>=54.

**Package:** `monotonic-nn` (pip install monotonic-nn).

### Option 3: Reduced Gompertz with Severity (RECOMMENDED)

Following Vaghi et al. (2020), the growth rate is a function of severity:

$$V(t) = V_0 \exp\!\left(\frac{\alpha(s_i)}{\beta}\bigl(1 - e^{-\beta t}\bigr)\right)$$

where $\alpha(s_i) = \alpha_0 + \alpha_1 \cdot s_i$ ($\alpha_1 > 0$ so growth rate increases with severity) and $\beta > 0$ is the shared deceleration rate.

In quantile space:

$$q_{ij} = \hat{F}\!\left(\frac{\alpha(s_i)}{\beta}\bigl(1 - e^{-\beta t_{ij}}\bigr)\right)$$

**Parameters:** 3 population ($\alpha_0, \alpha_1, \beta$) + N severity values. Strongest biological grounding.

**Recommendation:** Use Option 3 as primary, Option 1 as ablation. Skip Option 2 unless N >= 54.

---

## Severity Estimation

### Approach A: Joint MLE (Recommended)

Treat $\{s_i\}$ and $\boldsymbol{\theta}$ as unknowns estimated jointly:

$$\min_{\boldsymbol{\theta},\, \{s_i\}} \sum_{i=1}^{N} \sum_{j=1}^{n_i} \bigl(q_{ij} - g(s_i, t_{ij};\, \boldsymbol{\theta})\bigr)^2 + \lambda \|\boldsymbol{\theta}\|^2$$

subject to $s_i \in [0, 1]$, $w_1, w_2, \alpha_1 > 0$.

**Parameter count:** $|\boldsymbol{\theta}|$ + N. For N=31 and 3 population params: 34 parameters from ~112 observations. Tight but feasible because each $s_i$ is individually constrained by that patient's 2--5 observations.

**Implementation:**
```python
def optimize_severity_model(
    t_quantiles: np.ndarray,       # [N_obs]
    q_growths: np.ndarray,          # [N_obs]
    patient_indices: np.ndarray,    # [N_obs] -> patient index
    growth_function: str = "gompertz_reduced",
    lambda_reg: float = 0.01,
    n_restarts: int = 10,
    max_iter: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint optimization of theta and s_i values.

    Uses scipy.optimize.minimize with L-BFGS-B.
    Severity values bounded to [0, 1] via bounds parameter.

    Returns:
        (theta, severities): Population params and per-patient severities.
    """
```

**Package:** `scipy.optimize.minimize` with `method='L-BFGS-B'`.

### Approach B: Bayesian via pymc/numpyro (If Posterior Uncertainty Needed)

For credible intervals on severity (useful for thesis presentation but not required for LOPO comparison):

```python
# Using pymc v5
import pymc as pm
with pm.Model():
    s = pm.Beta("s", alpha=2, beta=2, shape=N)
    alpha_0 = pm.Normal("alpha_0", mu=0, sigma=1)
    alpha_1 = pm.HalfNormal("alpha_1", sigma=1)
    beta = pm.HalfNormal("beta", sigma=1)
    # ... growth function + likelihood
```

**When to use:** Only if posterior uncertainty on individual severities is needed for the thesis defense. Joint MLE is sufficient for LOPO-CV comparison.

### Approach C: Two-Step (Initialization Only)

1. Set initial $\hat{s}_i$ from baseline volume percentile: $\hat{s}_i^{(0)} = \hat{F}(\log(V_{i0} + 1))$.
2. Fit $\boldsymbol{\theta}$ given $\hat{s}^{(0)}$.
3. Use result as initialization for Approach A.

**Purpose:** Warm-start for joint optimization. Not a standalone approach.

---

## Test-Time Severity Estimation

At test time, given a single baseline scan, estimate severity from accessible features:

$$\hat{s}_{\text{new}} = \sigma\!\bigl(\mathbf{w}^T \mathbf{x}_{\text{baseline}} + b\bigr)$$

where $\mathbf{x}_{\text{baseline}}$ includes:
- $\log(V_{\text{baseline}} + 1)$: baseline tumor volume
- Age at baseline (when available)
- Sex (when available)
- Sphericity (from `compute_sphericity()` in `semantic_features.py`)

This is a 3--4 feature logistic regression, well within the budget at N=31.

**Training:** Fit on training patients' estimated severities $\{\hat{s}_i\}$ from joint optimization.

**LOPO-CV integration:** At each LOPO fold, the held-out patient's severity must be estimated from the severity regression head (fitted on training patients only), NOT from the joint optimization. Using joint-optimized severity for the held-out patient constitutes data leakage.

```python
class SeverityRegressionHead:
    """Predict severity from baseline features via logistic regression.

    Args:
        feature_names: List of feature names to extract from PatientTrajectory.
    """
    def fit(self, severities: np.ndarray, features: np.ndarray) -> None:
        """Fit logistic regression: s = sigmoid(w^T x + b)."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict severity for new patients. Returns values in [0, 1]."""
```

---

## Input Contract

```python
# From Stage 1 pipeline
patient_trajectories: list[PatientTrajectory]
# Each with observations = log(V + 1), shape [n_i, 1]

# Baseline features for severity regression
baseline_features: np.ndarray  # [N_patients, n_features]
feature_names: list[str]       # e.g. ["log_volume", "sphericity"]

# Stage 1 results (for comparison)
stage1_per_patient_errors: dict[str, np.ndarray]  # model_name -> errors [N]
stage1_best_r2: float
```

## Output Contract

```python
# Severity model results
severity_results: dict = {
    "model_name": str,                      # e.g. "SeverityGompertz"
    "growth_function": str,                 # "gompertz_reduced" | "weighted_sigmoid"
    "population_params": dict[str, float],  # theta values
    "severities": dict[str, float],         # patient_id -> s_i
    "severity_stats": {
        "mean": float, "std": float, "min": float, "max": float,
    },
    "lopo_metrics": {
        "last_from_rest/r2_log": float,
        "last_from_rest/mae_log": float,
        "last_from_rest/calibration_95": float,
        # ... same metric set as Stage 1
    },
    "bootstrap_ci": {
        "r2_log_point": float,
        "r2_log_lower_95": float,
        "r2_log_upper_95": float,
    },
    "severity_regression": {
        "features_used": list[str],
        "loo_r2": float,                    # Leave-one-out R^2 for s prediction
        "coefficients": dict[str, float],
    },
    "comparison_to_stage1": {
        "delta_r2": float,                  # Stage 2 R^2 - Stage 1 best R^2
        "permutation_p_value": float,       # Paired permutation test
    },
}

# Output files
# experiments/severity_model/results/
#   severity_model_results.json
#   lopo_results.json
#   severity_distribution.png
#   severity_vs_volume.png
#   predicted_vs_actual_quantiles.png
```

---

## Code Requirements

### New Files

1. **`src/growth/models/growth/quantile_transform.py`** -- Quantile transformation utilities.
   ```python
   def compute_time_quantiles(trajectories: list[PatientTrajectory]) -> dict[str, np.ndarray]:
       """Map raw times to empirical CDF quantiles in [0, 1].

       Returns dict mapping patient_id to t_quantile array [n_i].
       Baseline (t=0) maps to 0 exactly.
       """

   def compute_growth_quantiles(trajectories: list[PatientTrajectory]) -> dict[str, np.ndarray]:
       """Map delta-log-volumes to empirical growth quantiles in [0, 1].

       Returns dict mapping patient_id to q_growth array [n_i].
       Baseline maps to 0 exactly.
       """

   def quantile_to_log_volume(
       q_predicted: np.ndarray,
       empirical_cdf: np.ndarray,
       empirical_values: np.ndarray,
   ) -> np.ndarray:
       """Inverse quantile transform: q -> delta_log_volume.

       Uses linear interpolation on the empirical CDF.
       """
   ```

2. **`src/growth/models/growth/severity_model.py`** -- Core severity growth model.
   ```python
   class SeverityGrowthModel(GrowthModel):
       """NLME with latent severity and monotonic growth function.

       Extends GrowthModel interface. The fit() method jointly optimizes
       population parameters and per-patient severities. The predict()
       method estimates held-out severity from the regression head.

       Args:
           growth_function: "gompertz_reduced" | "weighted_sigmoid"
           lambda_reg: L2 regularization on population params.
           n_restarts: Random restarts for L-BFGS-B.
           max_iter: Maximum optimizer iterations.
           severity_features: Feature names for severity regression.
       """

       def fit(self, patients: list[PatientTrajectory]) -> FitResult:
           """Joint optimization of theta and s_i values.

           Steps:
             1. Compute quantile transforms from patient trajectories
             2. Initialize s_i from baseline volume percentile
             3. Joint L-BFGS-B optimization
             4. Fit severity regression head on estimated s_i
           """

       def predict(
           self, patient: PatientTrajectory,
           t_pred: np.ndarray,
           n_condition: int | None = None,
       ) -> PredictionResult:
           """Predict using severity estimated from regression head.

           Does NOT use joint-optimized severity for new patients.
           Predicts in quantile space, then inverse-transforms to log-volume.
           """
   ```

3. **`experiments/severity_model/run_severity.py`** -- Orchestrator.
   ```python
   def main():
       """Run severity model under LOPO-CV and compare to Stage 1."""
       # 1. Load trajectories (same as Stage 1)
       # 2. Fit severity model under LOPO-CV
       # 3. Compute bootstrap CIs
       # 4. Compare to Stage 1 via paired permutation test
       # 5. Generate figures
       # 6. Save results
   ```

4. **`experiments/severity_model/config.yaml`** -- Configuration.

### Modifications to Existing Files

- `src/growth/models/growth/__init__.py`: Add `SeverityGrowthModel` to exports.
- `src/growth/evaluation/lopo_evaluator.py`: The `LOPOEvaluator` already supports any `GrowthModel` subclass. The `SeverityGrowthModel.fit()` must handle severity re-estimation internally per fold.

---

## Configuration

```yaml
# experiments/severity_model/config.yaml
experiment:
  name: stage2_severity_model
  seed: 42

paths:
  mengrowth_h5: /path/to/MenGrowth.h5
  output_dir: experiments/severity_model/results
  stage1_results: experiments/segment_based_approach/results

severity:
  growth_function: gompertz_reduced  # gompertz_reduced | weighted_sigmoid
  optimization:
    method: lbfgsb
    max_iter: 5000
    n_restarts: 10
    lambda_reg: 0.01
    bounds_severity: [0.0, 1.0]
  severity_estimation:
    features: [log_volume, sphericity]  # + age, sex when available
    model: logistic_regression

quantile_transform:
  time_method: rank_average      # scipy.stats.rankdata(method='average')
  growth_variable: delta_log1p   # log(V_j+1) - log(V_0+1)

volume:
  transform: log1p

patients:
  exclude: [MenGrowth-0028]
  min_timepoints: 2

bootstrap:
  enabled: true
  n_samples: 2000
  method: bca
  seed: 42

comparison:
  permutation_test_n: 10000
```

---

## LOPO-CV Protocol Detail

The severity model requires special handling under LOPO-CV to avoid data leakage:

```
For each fold k (held-out patient p_k):
  1. Training patients = all patients except p_k
  2. Compute quantile transforms on training patients ONLY
     - t_quantiles from training time intervals
     - q_growths from training growth values
  3. Joint-optimize (theta, {s_i}_{i != k}) on training patients
  4. Fit severity regression head: s ~ f(baseline_features) on training patients
  5. For held-out patient p_k:
     a. Extract baseline features
     b. Predict severity: s_hat_k = regression_head.predict(features_k)
     c. Transform p_k's times using TRAINING quantile mapping
     d. Predict growth using g(s_hat_k, t; theta)
     e. Inverse-transform predicted quantiles to log-volume space
  6. Record predictions and actual values
```

**Critical:** Step 5b uses the regression head, NOT the joint-optimized severity. This prevents data leakage.

---

## Verification Tests

```
S2-T1: Quantile transform produces valid values [BLOCKING]
  - Assert all t_quantile in [0, 1]
  - Assert all q_growth in [0, 1]
  - Assert baseline t = 0 and baseline q = 0
  - Assert monotonicity: if Delta_t_a < Delta_t_b then t_a <= t_b
  Recovery: Check rankdata method; verify baseline handling

S2-T2: Joint optimization converges [BLOCKING]
  - Loss decreases over iterations
  - Final loss < initial loss
  - Assert theta values are finite
  - Assert all s_i in [0, 1]
  - Run with 3 random seeds; assert results are consistent (std/mean < 0.1)
  Recovery: Increase n_restarts to 20; try different initialization

S2-T3: Monotonicity constraints satisfied [BLOCKING]
  - Evaluate g at a 50x50 grid of (s, t) in [0, 1]^2
  - Assert partial_g/partial_t >= -1e-6 at all grid points
  - Assert partial_g/partial_s >= -1e-6 at all grid points
  - Check numerically via finite differences (step = 0.01)
  Recovery: Add monotonicity penalty to loss; use projected gradient

S2-T4: Boundary condition holds exactly [BLOCKING]
  - Assert g(s, 0) = 0 for s in {0, 0.25, 0.5, 0.75, 1.0}
  - Assert |g(s, 0)| < 1e-10
  Recovery: Verify multiplication by t (Option 1) or exp-minus-1 structure (Option 3)

S2-T5: Severity distribution has spread [BLOCKING]
  - Assert std(s_i) > 0.15 (not collapsed to a single point)
  - Assert min(s_i) < 0.3 and max(s_i) > 0.7 (uses range)
  - Plot histogram of estimated severities
  Recovery: Reduce lambda_reg; check if growth function is too flexible

S2-T6: Severity correlates with clinical proxies [DIAGNOSTIC]
  - Compute Spearman(s_i, log_baseline_volume) > 0.3
  - Compute Spearman(s_i, observed_growth_rate) > 0.5
  - Report correlations with all available features
  Note: DIAGNOSTIC -- weak correlation may indicate severity captures
    non-obvious information, which is scientifically interesting

S2-T7: LOPO-CV R^2_log >= Stage 1 best [BLOCKING for Stage 2 justification]
  - Run full LOPO-CV with severity model
  - Compare R^2_log to Stage 1 best model
  - Compute paired permutation test p-value
  - If p > 0.05, Stage 2 does NOT justify its added complexity
  Recovery: Try alternative growth function; increase n_restarts

S2-T8: Severity regression head from baseline features [DIAGNOSTIC]
  - Leave-one-out R^2 for severity prediction from baseline features
  - Assert R^2 > 0 (severity is predictable from baseline)
  - Report which features are most predictive
  Note: DIAGNOSTIC -- poor severity prediction means the model cannot
    individualize at test time, but the severity construct may still be valid
```

---

## References

- Vaghi, C. et al. "Population modeling of tumor growth curves and the reduced Gompertz model," *PLOS Computational Biology*, 2020.
- Proust-Lima, C. et al. "Joint latent class models for longitudinal and time-to-event data," *Statistical Methods in Medical Research*, 2014.
- Proust-Lima, C. et al. "Estimation of extended mixed models using latent classes and latent processes: the R package lcmm," *Journal of Statistical Software*, 2023.
- Runje, D. & Shankaranarayana, S. M. "Constrained Monotonic Neural Networks," *ICML*, 2023.
- Riihimaki, J. & Vehtari, A. "Gaussian processes with monotonicity information," *AISTATS*, 2010.
- Baker, F. B. & Kim, S.-H. *Item Response Theory: Parameter Estimation Techniques*, CRC Press, 2004.
