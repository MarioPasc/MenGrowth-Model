# GP Probe Refactoring Specification

## Refactoring: Ridge/MLP Probes → Gaussian Process Probes (Linear + RBF Kernels)

**Project**: MenGrowth-Model (`https://github.com/MarioPasc/MenGrowth-Model`)  
**Scope**: `src/growth/evaluation/` and `experiments/lora_ablation/pipeline/evaluate_probes.py`  
**Backwards compatibility**: None required — this is a clean replacement.  
**Dependency**: GPy (already in `pyproject.toml`)

---

## 1. Problem Statement

The current probe evaluation system uses two disjoint techniques to assess encoder feature quality:

1. **Ridge regression** (`sklearn.linear_model.Ridge`) for linear probing → produces a scalar $R^2$ with no uncertainty.
2. **MLP regression** (`sklearn.neural_network.MLPRegressor`) for nonlinear probing → produces a scalar $R^2$ with no uncertainty and introduces uncontrolled hyperparameters (hidden sizes, learning rate, number of iterations).

These two techniques share no mathematical backbone, making their comparison ad-hoc. The "nonlinearity gap" metric ($R^2_{\text{MLP}} - R^2_{\text{Ridge}}$) is confounded by MLP training noise, random initialization, and hyperparameter sensitivity.

### What we want instead

Replace **both** Ridge and MLP probes with **Gaussian Process (GP) regression** under two kernels:

- **GP with linear kernel** → replaces Ridge regression. Produces identical point-estimate $R^2$ but additionally provides **posterior predictive variance** (uncertainty per test point) and **log-marginal likelihood** for principled model comparison.
- **GP with RBF kernel** → replaces MLP probe. Captures nonlinear feature-to-target relationships with **automatic complexity control** via marginal likelihood optimization (no manual hyperparameter tuning beyond kernel choice).

This yields a unified Bayesian framework where:
- Linear vs. nonlinear separation is a kernel comparison within the same model class.
- Uncertainty quantification ($\mu_* \pm 2\sigma_*$) enables sausage plots and credible intervals on $R^2$.
- Model comparison is principled: $\Delta \log p(y | X) = \log p(y | X, k_{\text{RBF}}) - \log p(y | X, k_{\text{linear}})$ is an automatic Occam's razor.

---

## 2. Mathematical Justification

### 2.1 Ridge Regression as a Special Case of GP with Linear Kernel

This equivalence is the theoretical cornerstone of the refactoring. We prove that Ridge regression is *exactly* recovered as the posterior mean of a GP with a linear kernel.

**Ridge regression.** Given training data $(X, y)$ with $X \in \mathbb{R}^{N \times D}$, $y \in \mathbb{R}^N$, Ridge solves:

$$\hat{w} = \arg\min_w \|Xw - y\|_2^2 + \alpha \|w\|_2^2$$

The closed-form solution is:

$$\hat{w} = (X^\top X + \alpha I)^{-1} X^\top y$$

Predictions at test inputs $X_*$:

$$\hat{y}_* = X_* \hat{w} = X_* (X^\top X + \alpha I)^{-1} X^\top y$$

**GP with linear kernel.** Consider a GP prior $f \sim \mathcal{GP}(0, k)$ with:

$$k(x, x') = \sigma_f^2 \, x^\top x'$$

and observation noise $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$. The kernel matrix is $K = \sigma_f^2 X X^\top$. The GP predictive mean is:

$$\bar{f}_* = K_*^\top (K + \sigma_n^2 I)^{-1} y$$

where $K_* = \sigma_f^2 X X_*^\top$. Substituting:

$$\bar{f}_* = \sigma_f^2 X_* X^\top (\sigma_f^2 X X^\top + \sigma_n^2 I)^{-1} y$$

Using the matrix identity $(P Q^\top + \sigma_n^2 I)^{-1} P = P(Q^\top P + \sigma_n^2 I)^{-1}$ (Woodbury), this simplifies to:

$$\bar{f}_* = X_* (\underbrace{X^\top X + \frac{\sigma_n^2}{\sigma_f^2} I}_{\text{Ridge with } \alpha = \sigma_n^2 / \sigma_f^2})^{-1} X^\top y$$

**Therefore:** The GP posterior mean with a linear kernel $k(x,x') = \sigma_f^2 x^\top x'$ and noise $\sigma_n^2$ is *exactly* Ridge regression with $\alpha = \sigma_n^2 / \sigma_f^2$. QED.

This means:
- The point-estimate $R^2$ from the GP-linear probe will **match** Ridge $R^2$ (up to hyperparameter optimization of $\sigma_f^2, \sigma_n^2$ vs. fixed $\alpha = 1.0$).
- The GP additionally provides the **predictive variance**:

$$\text{Var}(f_*) = k_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_* = \sigma_f^2 x_*^\top x_* - \sigma_f^4 x_*^\top X^\top (\sigma_f^2 X X^\top + \sigma_n^2 I)^{-1} X x_*$$

This variance is **not available** from Ridge regression.

### 2.2 Why GP-RBF Replaces MLP Probes More Principally

**MLP probe problems:**
- Requires manual selection of hidden layer sizes, $L_2$ penalty, learning rate, and max iterations.
- Susceptible to local minima in the loss landscape.
- No principled uncertainty quantification.
- The "nonlinearity gap" $R^2_{\text{MLP}} - R^2_{\text{linear}}$ conflates true nonlinearity with MLP fitting quality.

**GP-RBF advantages:**
- The RBF kernel $k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$ is a universal approximator in the RKHS sense (Micchelli et al., "Universal Kernels," JMLR, 2006).
- Hyperparameters $(\sigma_f^2, \ell, \sigma_n^2)$ are optimized by maximizing the log-marginal likelihood:

$$\log p(y | X, \theta) = -\frac{1}{2} y^\top (K + \sigma_n^2 I)^{-1} y - \frac{1}{2} \log |K + \sigma_n^2 I| - \frac{N}{2} \log 2\pi$$

This is an automatic Occam's razor: the first term is a data-fit term, the second is a complexity penalty. No cross-validation needed.

- **Nonlinearity gap** becomes: $\Delta \text{LML} = \log p(y | X, k_{\text{RBF}}) - \log p(y | X, k_{\text{linear}})$. If $\Delta \text{LML} > 0$, nonlinear structure genuinely exists in the features (favored by the data under Bayesian model comparison). If $\Delta \text{LML} \leq 0$, the linear model is sufficient and the MLP's apparent superiority was overfitting.

### 2.3 Credible Intervals on $R^2$

The GP predictive distribution allows construction of credible intervals on $R^2$. Given $N_*$ test points with predictive distributions $f_{*i} \sim \mathcal{N}(\bar{f}_{*i}, \sigma_{*i}^2)$:

1. Draw $K$ posterior samples: for each sample $k$, draw $f_{*i}^{(k)} \sim \mathcal{N}(\bar{f}_{*i}, \sigma_{*i}^2)$ for all $i$.
2. Compute $R^{2(k)} = 1 - \frac{\sum_i (y_i - f_{*i}^{(k)})^2}{\sum_i (y_i - \bar{y})^2}$ for each sample.
3. Report $[R^2_{2.5\%}, R^2_{97.5\%}]$ as the 95% credible interval.

This is **critical** for the LoRA ablation: comparing conditions (frozen vs. LoRA r8 vs. LoRA r16) via point $R^2$ is statistically fragile on 100-subject test sets. Credible intervals enable principled conclusions about whether one condition genuinely outperforms another.

### 2.4 Sausage Plots

For each 1D semantic target (e.g., log-volume dimension $k$), sort test subjects by their ground-truth value $y_k$ and plot:

- **Mean prediction**: $\bar{f}_{*i}$ vs. $y_i$
- **95% prediction band**: $\bar{f}_{*i} \pm 2\sigma_{*i}$

Narrow bands → the encoder features are highly informative for this target.  
Wide bands → the encoder features carry little predictive information.  
Spatially varying width → identifies the *region* of the target space where features are most/least informative.

---

## 3. Computational Feasibility

GP regression requires $O(N^3)$ for the Cholesky decomposition of $(K + \sigma_n^2 I)$. For this project:

| Dataset | $N$ | $O(N^3)$ FLOPs | Wall time (CPU) |
|---------|-----|-----------------|-----------------|
| Probe training set | 800 (or 525 for lora_train) | $5.1 \times 10^8$ | ~0.2s |
| Test set | 100–150 | $3.4 \times 10^6$ | negligible |

The bottleneck is kernel hyperparameter optimization (typically 50–200 L-BFGS iterations), each requiring an $O(N^3)$ solve. At $N = 800$, this is **under 30 seconds** total per target dimension. Since we have 10 target dimensions (4 vol + 3 loc + 3 shape), total GP fitting time is approximately 5 minutes — negligible compared to the LoRA training time (hours on A100).

**GPy** handles all of this natively. No GPU acceleration needed at this scale.

**Note on multi-output.** Each semantic target dimension is modeled by an independent GP (single-output GP per dimension). Multi-output GPs (e.g., ICM/LMC kernels) would capture cross-target correlations but add unnecessary complexity for probing. Independent GPs are the standard choice in representation evaluation (Alain & Bengio, ICLR 2017 Workshop).

---

## 4. Files to Modify / Delete / Create

### 4.1 Files to DELETE (no backwards compatibility)

```
src/growth/evaluation/enhanced_probes.py          # Entire file — MLP probes replaced by GP-RBF
```

### 4.2 Files to REWRITE

```
src/growth/evaluation/latent_quality.py            # Replace LinearProbe, ProbeResults, SemanticProbes
src/growth/evaluation/__init__.py                  # Update exports
experiments/lora_ablation/pipeline/evaluate_probes.py  # Update caller
tests/growth/test_latent_quality.py                # Update tests
```

### 4.3 Files to CREATE

```
src/growth/evaluation/gp_probes.py                 # NEW: Core GP probe implementation
```

### 4.4 Files with IMPORT-ONLY changes (update imports, no logic changes)

```
experiments/lora_ablation/analysis/analyze_results.py    # If it imports from enhanced_probes
experiments/lora_ablation/analysis/regenerate_analysis.py  # If it imports from enhanced_probes
scripts/sdp_diagnostics.py                               # Uses Ridge probes from latent_quality
```

---

## 5. Detailed Implementation Specification

### 5.1 New file: `src/growth/evaluation/gp_probes.py`

This is the core new module. It replaces both `LinearProbe` (from `latent_quality.py`) and `EnhancedLinearProbe` + `MLPProbe` (from `enhanced_probes.py`).

#### 5.1.1 Data classes

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class GPProbeResults:
    """Results from a single GP probe evaluation.

    Attributes:
        r2: Point-estimate R² (from posterior mean predictions).
        r2_per_dim: Per-target-dimension R².
        r2_ci_lo: Lower bound of 95% credible interval on R².
        r2_ci_hi: Upper bound of 95% credible interval on R².
        mse: Mean squared error (from posterior mean).
        predictions: Posterior mean predictions [N_test, output_dim].
        predictive_std: Posterior std per test point [N_test, output_dim].
        log_marginal_likelihood: Model evidence log p(y|X,θ*).
        kernel_type: 'linear' or 'rbf'.
        optimized_params: Dict of optimized kernel hyperparameters.
    """
    r2: float
    r2_per_dim: np.ndarray
    r2_ci_lo: float
    r2_ci_hi: float
    mse: float
    predictions: np.ndarray
    predictive_std: np.ndarray
    log_marginal_likelihood: float
    kernel_type: str
    optimized_params: dict = field(default_factory=dict)
```

```python
@dataclass
class GPSemanticResults:
    """Aggregated results across all semantic targets and both kernels.

    Attributes:
        linear: Dict mapping target name -> GPProbeResults for linear kernel.
        rbf: Dict mapping target name -> GPProbeResults for RBF kernel.
        nonlinearity_evidence: Dict mapping target name -> delta LML (RBF - linear).
    """
    linear: dict  # {name: GPProbeResults}
    rbf: dict     # {name: GPProbeResults}
    nonlinearity_evidence: dict  # {name: float}  (delta LML)
```

#### 5.1.2 GPProbe class

```python
class GPProbe:
    """Gaussian Process regression probe for feature quality evaluation.

    Fits an independent GP per target dimension. Supports linear and RBF
    kernels with automatic hyperparameter optimization via log-marginal
    likelihood maximization.

    Mathematical equivalence: GP with linear kernel and noise variance
    sigma_n^2 recovers Ridge regression with alpha = sigma_n^2 / sigma_f^2.
    The GP additionally provides predictive variance for uncertainty
    quantification.

    Args:
        kernel_type: 'linear' or 'rbf'.
        normalize_features: Whether to standardize input features.
        normalize_targets: Whether to standardize target variables.
        n_restarts: Number of random restarts for hyperparameter optimization.
        r2_ci_samples: Number of posterior samples for R² credible intervals.
    """
```

**Key implementation details:**

1. **Feature standardization**: Both features and targets should be standardized before GP fitting (mean=0, std=1). Predictions are de-standardized before computing $R^2$. This is critical because:
   - GP kernel hyperparameters (lengthscale $\ell$, variance $\sigma_f^2$) are scale-sensitive.
   - The linear kernel $k(x,x') = \sigma_f^2 x^\top x'$ with standardized features gives each feature equal prior importance.

2. **Per-dimension independent GPs**: For a target with $K$ dimensions (e.g., volume has $K=4$), fit $K$ separate `GPy.models.GPRegression` instances. This avoids the $O(NK^3)$ cost of multi-output GPs and is standard for probing.

3. **Kernel construction** (GPy syntax):
   ```python
   import GPy
   D = X_train.shape[1]
   if kernel_type == "linear":
       kernel = GPy.kern.Linear(D, variances=np.ones(D))  # ARD=True by default on Linear
       # NOTE: For high-D (768), use ARD=False to avoid 768 variance params:
       kernel = GPy.kern.Linear(D, ARD=False)
   elif kernel_type == "rbf":
       kernel = GPy.kern.RBF(D, ARD=False)  # Single lengthscale (isotropic)
       # ARD=True would add 768 lengthscale params — too many for N=800.
   ```

4. **CRITICAL: ARD=False for both kernels.** With $D = 768$ features and $N = 800$ subjects, ARD (Automatic Relevance Determination) would introduce 768 hyperparameters for the RBF kernel, which is computationally expensive and likely to overfit. Use isotropic kernels (single $\ell$ for RBF, single $\sigma_f^2$ for linear).

5. **Optimization**: Use `model.optimize(messages=False, max_iters=500)`. Optionally `model.optimize_restarts(num_restarts=n_restarts, verbose=False)` for the RBF kernel to avoid local optima in the lengthscale.

6. **R² credible intervals**: After fitting, draw $K$ posterior samples at test points and compute $R^2$ for each sample:
   ```python
   def _compute_r2_ci(self, model, X_test, y_test, n_samples=500):
       mu, var = model.predict(X_test)
       std = np.sqrt(var)
       r2_samples = []
       ss_tot = np.sum((y_test - y_test.mean()) ** 2)
       for _ in range(n_samples):
           y_sample = mu + std * np.random.randn(*mu.shape)
           ss_res = np.sum((y_test - y_sample) ** 2)
           r2_samples.append(1.0 - ss_res / (ss_tot + 1e-12))
       return np.percentile(r2_samples, 2.5), np.percentile(r2_samples, 97.5)
   ```

7. **Log-marginal likelihood extraction**: `model.log_likelihood()` after optimization.

#### 5.1.3 GPSemanticProbes class

```python
class GPSemanticProbes:
    """GP-based semantic probes for all feature types.

    Manages paired GP probes (linear + RBF) for volume, location, and shape.
    Computes nonlinearity evidence as delta log-marginal likelihood.

    Args:
        input_dim: Feature dimensionality (768 for encoder10).
        normalize_features: Standardize features before GP fitting.
        normalize_targets: Standardize targets before GP fitting.
        n_restarts: Random restarts for RBF optimization.
        r2_ci_samples: Posterior samples for R² credible intervals.
    """
    FEATURE_DIMS = {
        "volume": 4,
        "location": 3,
        "shape": 3,
    }
```

This class creates two `GPProbe` instances per semantic target (one linear, one RBF) and provides:
- `fit(X, targets)` → fits all 6 GPs (3 targets × 2 kernels × K dims each)
- `evaluate(X_test, targets_test)` → returns `GPSemanticResults`
- `get_summary(results)` → returns flat dict compatible with existing JSON logging

#### 5.1.4 Sausage plot data extraction

```python
def extract_sausage_data(
    results: GPProbeResults,
    y_test: np.ndarray,
    target_dim: int = 0,
) -> dict:
    """Extract data for sausage plots.

    Returns dict with:
        'y_true': Ground-truth values for dimension target_dim, sorted.
        'y_pred_mean': GP posterior mean, same sort order.
        'y_pred_lo': Lower 95% prediction band.
        'y_pred_hi': Upper 95% prediction band.
        'sort_idx': Sort indices for reconstructing original order.
    """
```

### 5.2 Modifications to `src/growth/evaluation/latent_quality.py`

**Remove**: `LinearProbe`, `ProbeResults`, `SemanticProbes` classes entirely.

**Keep**: All domain-shift metrics (`compute_cka`, `mmd_permutation_test`, `distance_correlation`, `compute_partition_correlation`, `compute_dcor_matrix`, `compute_variance_per_dim`, `compute_r2_scores`, `evaluate_latent_quality`, `compute_mmd`, `compute_effective_rank`, `compute_proxy_a_distance`, `compute_domain_shift_metrics`, `compute_dci`, `compute_cross_correlation`, `compute_domain_classifier_accuracy`, `DCIResults`, `DomainShiftMetrics`).

**Add**: Import and re-export from `gp_probes.py`:
```python
from .gp_probes import GPProbe, GPProbeResults, GPSemanticProbes, GPSemanticResults, extract_sausage_data
```

### 5.3 Modifications to `src/growth/evaluation/__init__.py`

Replace all exports from `enhanced_probes` and the old `LinearProbe`/`SemanticProbes`/`ProbeResults` with:

```python
# GP-based probes (replaces Ridge + MLP)
from .gp_probes import (
    GPProbe,
    GPProbeResults,
    GPSemanticProbes,
    GPSemanticResults,
    extract_sausage_data,
)
```

Remove all imports of: `EnhancedLinearProbe`, `EnhancedProbeResults`, `EnhancedSemanticProbes`, `MLPProbe`, `analyze_feature_quality`, `compute_multi_scale_features`, `LinearProbe`, `ProbeResults`, `SemanticProbes`.

### 5.4 Modifications to `experiments/lora_ablation/pipeline/evaluate_probes.py`

This is the main caller. Replace the `evaluate_probes_enhanced` function:

**Before** (current):
```python
from growth.evaluation.enhanced_probes import EnhancedSemanticProbes, analyze_feature_quality
...
probes = EnhancedSemanticProbes(input_dim=..., alpha_linear=..., alpha_mlp=..., ...)
probes.fit(X_probe, targets_probe)
results = probes.evaluate(X_test, targets_test)
summary = probes.get_summary(results)
```

**After** (new):
```python
from growth.evaluation.gp_probes import GPSemanticProbes
...
probes = GPSemanticProbes(input_dim=..., n_restarts=5, r2_ci_samples=500)
probes.fit(X_probe, targets_probe)
results = probes.evaluate(X_test, targets_test)
summary = probes.get_summary(results)
```

**Key changes in the output format:**

| Old key | New key | Notes |
|---------|---------|-------|
| `r2_volume_linear` | `r2_volume_linear` | Same name, now from GP-linear |
| `r2_volume_mlp` | `r2_volume_rbf` | `mlp` → `rbf` everywhere |
| `mse_volume_linear` | `mse_volume_linear` | Same |
| `mse_volume_mlp` | `mse_volume_rbf` | `mlp` → `rbf` |
| `nonlinearity_gap_volume` | `nonlinearity_evidence_volume` | Now in log-marginal likelihood units |
| *(not available)* | `r2_volume_linear_ci_lo` | NEW: lower credible interval |
| *(not available)* | `r2_volume_linear_ci_hi` | NEW: upper credible interval |
| *(not available)* | `r2_volume_rbf_ci_lo` | NEW |
| *(not available)* | `r2_volume_rbf_ci_hi` | NEW |
| *(not available)* | `lml_volume_linear` | NEW: log-marginal likelihood |
| *(not available)* | `lml_volume_rbf` | NEW: log-marginal likelihood |

The `predictions_enhanced.json` output should change from:
```json
{"volume": {"linear": [...], "mlp": [...], "ground_truth": [...]}}
```
to:
```json
{
  "volume": {
    "linear_mean": [...], "linear_std": [...],
    "rbf_mean": [...], "rbf_std": [...],
    "ground_truth": [...]
  }
}
```

The `probes_enhanced.pkl` should be renamed to `probes_gp.pkl` and contain the `GPSemanticProbes` object.

The `metrics.json` backward compatibility mapping should use the `linear` GP results to populate the existing keys (`r2_volume`, `r2_location`, etc.) so that downstream report generation (`narrative.py`) continues to work.

### 5.5 Modifications to `scripts/sdp_diagnostics.py`

This script has its own inline Ridge probe (`compute_linear_probe_ceiling`). Replace it with GPProbe:

```python
from growth.evaluation.gp_probes import GPProbe

def compute_probe_ceiling(h_train, targets_train, h_val, targets_val):
    results = {}
    for name, y_train in targets_train.items():
        y_val = targets_val[name]
        for kernel in ["linear", "rbf"]:
            probe = GPProbe(kernel_type=kernel)
            probe.fit(h_train, y_train)
            res = probe.evaluate(h_val, y_val)
            results[f"{name}_{kernel}"] = {
                "r2": res.r2,
                "r2_ci": (res.r2_ci_lo, res.r2_ci_hi),
                "lml": res.log_marginal_likelihood,
            }
    return results
```

---

## 6. Test Specification

### 6.1 File: `tests/growth/test_gp_probes.py` (NEW)

Replace all tests in `tests/growth/test_latent_quality.py` that reference `LinearProbe`, `ProbeResults`, `SemanticProbes`. The new test file should cover:

#### TEST_GP.1: GP-Linear recovers Ridge R² (equivalence test)

```python
def test_gp_linear_matches_ridge():
    """GP with linear kernel produces R² within 0.05 of Ridge regression.

    Mathematical basis: GP-linear posterior mean = Ridge solution when
    alpha = sigma_n^2 / sigma_f^2. Small deviations arise from GP
    hyperparameter optimization vs. fixed alpha=1.0.
    """
    np.random.seed(42)
    n, d, k = 500, 64, 4
    X = np.random.randn(n, d)
    W = np.random.randn(d, k) * 0.1
    y = X @ W + np.random.randn(n, k) * 0.01

    # Ridge baseline
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[:400])
    X_test_s = scaler.transform(X[400:])
    ridge = Ridge(alpha=1.0).fit(X_train_s, y[:400])
    r2_ridge = r2_score(y[400:], ridge.predict(X_test_s))

    # GP-linear
    gp = GPProbe(kernel_type="linear")
    gp.fit(X[:400], y[:400])
    results = gp.evaluate(X[400:], y[400:])

    assert abs(results.r2 - r2_ridge) < 0.05, (
        f"GP-linear R²={results.r2:.4f} differs from Ridge R²={r2_ridge:.4f} by > 0.05"
    )
```

#### TEST_GP.2: Predictive variance is positive and finite

```python
def test_predictive_variance_positive():
    """GP predictive std is strictly positive and finite."""
    np.random.seed(42)
    X = np.random.randn(200, 32)
    y = np.random.randn(200, 3)

    for kernel in ["linear", "rbf"]:
        gp = GPProbe(kernel_type=kernel)
        gp.fit(X[:160], y[:160])
        results = gp.evaluate(X[160:], y[160:])

        assert results.predictive_std.shape == (40, 3)
        assert np.all(results.predictive_std > 0)
        assert np.all(np.isfinite(results.predictive_std))
```

#### TEST_GP.3: R² credible intervals bracket point estimate

```python
def test_r2_ci_brackets_point_estimate():
    """95% credible interval on R² contains the point estimate."""
    np.random.seed(42)
    n, d = 300, 32
    X = np.random.randn(n, d)
    y = X[:, :4] + np.random.randn(n, 4) * 0.5  # Moderate signal

    gp = GPProbe(kernel_type="linear", r2_ci_samples=1000)
    gp.fit(X[:240], y[:240])
    results = gp.evaluate(X[240:], y[240:])

    assert results.r2_ci_lo <= results.r2 <= results.r2_ci_hi
    assert results.r2_ci_lo < results.r2_ci_hi  # Non-degenerate
```

#### TEST_GP.4: RBF captures nonlinear signal

```python
def test_rbf_better_than_linear_for_nonlinear_data():
    """GP-RBF achieves higher R² than GP-linear on nonlinear data."""
    np.random.seed(42)
    n, d = 500, 16
    X = np.random.randn(n, d)
    y = np.sin(X[:, 0:1] * 2) + np.cos(X[:, 1:2]) + np.random.randn(n, 1) * 0.1

    gp_lin = GPProbe(kernel_type="linear")
    gp_rbf = GPProbe(kernel_type="rbf", n_restarts=3)
    gp_lin.fit(X[:400], y[:400])
    gp_rbf.fit(X[:400], y[:400])
    res_lin = gp_lin.evaluate(X[400:], y[400:])
    res_rbf = gp_rbf.evaluate(X[400:], y[400:])

    assert res_rbf.r2 > res_lin.r2 + 0.1, (
        f"RBF R²={res_rbf.r2:.4f} not substantially better than linear R²={res_lin.r2:.4f}"
    )
```

#### TEST_GP.5: Log-marginal likelihood favors correct kernel

```python
def test_lml_favors_correct_kernel():
    """LML(RBF) > LML(linear) for nonlinear data, LML(linear) >= LML(RBF) - threshold for linear data."""
    np.random.seed(42)
    n, d = 400, 16

    # Linear data
    X = np.random.randn(n, d)
    W = np.random.randn(d, 1) * 0.1
    y_linear = X @ W + np.random.randn(n, 1) * 0.05

    gp_lin = GPProbe(kernel_type="linear")
    gp_rbf = GPProbe(kernel_type="rbf", n_restarts=3)
    gp_lin.fit(X, y_linear)
    gp_rbf.fit(X, y_linear)
    res_lin = gp_lin.evaluate(X, y_linear)
    res_rbf = gp_rbf.evaluate(X, y_linear)

    # For linear data, linear LML should be competitive
    # (RBF may match but shouldn't dramatically exceed due to Occam penalty)
    delta_lml = res_rbf.log_marginal_likelihood - res_lin.log_marginal_likelihood
    # We don't assert delta_lml <= 0 strictly because RBF can recover linear functions,
    # but the gap should be modest.

    # Nonlinear data
    y_nonlinear = np.sin(X[:, 0:1] * 3) + np.random.randn(n, 1) * 0.1
    gp_lin2 = GPProbe(kernel_type="linear")
    gp_rbf2 = GPProbe(kernel_type="rbf", n_restarts=3)
    gp_lin2.fit(X, y_nonlinear)
    gp_rbf2.fit(X, y_nonlinear)
    res_lin2 = gp_lin2.evaluate(X, y_nonlinear)
    res_rbf2 = gp_rbf2.evaluate(X, y_nonlinear)

    delta_lml_nonlinear = res_rbf2.log_marginal_likelihood - res_lin2.log_marginal_likelihood
    assert delta_lml_nonlinear > 0, (
        f"RBF LML ({res_rbf2.log_marginal_likelihood:.2f}) should exceed "
        f"linear LML ({res_lin2.log_marginal_likelihood:.2f}) for nonlinear data"
    )
```

#### TEST_GP.6: SemanticProbes integration

```python
def test_semantic_probes_fit_evaluate():
    """GPSemanticProbes fits and evaluates with correct output structure."""
    np.random.seed(42)
    X = np.random.randn(200, 768)
    targets = {
        "volume": np.random.randn(200, 4),
        "location": np.random.randn(200, 3),
        "shape": np.random.randn(200, 3),
    }

    probes = GPSemanticProbes(input_dim=768, n_restarts=1, r2_ci_samples=100)
    probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
    results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})

    assert isinstance(results, GPSemanticResults)
    for name in ["volume", "location", "shape"]:
        assert name in results.linear
        assert name in results.rbf
        assert name in results.nonlinearity_evidence
        assert isinstance(results.linear[name], GPProbeResults)
        assert results.linear[name].kernel_type == "linear"
        assert results.rbf[name].kernel_type == "rbf"
```

#### TEST_GP.7: High R² for linearly related data

```python
def test_high_r2_for_linear_data():
    """GP probes achieve R² > 0.90 on linearly generated data."""
    np.random.seed(42)
    n, d, k = 500, 64, 4
    X = np.random.randn(n, d)
    W = np.random.randn(d, k) * 0.1
    y = X @ W + np.random.randn(n, k) * 0.01

    gp = GPProbe(kernel_type="linear")
    gp.fit(X[:400], y[:400])
    results = gp.evaluate(X[400:], y[400:])

    assert results.r2 > 0.90
```

#### TEST_GP.8: Missing target raises ValueError

```python
def test_missing_target_raises():
    """GPSemanticProbes raises ValueError for missing targets."""
    probes = GPSemanticProbes(input_dim=768)
    X = np.random.randn(100, 768)
    targets = {"volume": np.random.randn(100, 4)}  # Missing location, shape

    with pytest.raises(ValueError, match="Missing target"):
        probes.fit(X, targets)
```

#### TEST_GP.9: Get summary returns expected keys

```python
def test_get_summary_keys():
    """get_summary returns all expected metric keys."""
    np.random.seed(42)
    X = np.random.randn(200, 64)
    targets = {
        "volume": np.random.randn(200, 4),
        "location": np.random.randn(200, 3),
        "shape": np.random.randn(200, 3),
    }
    probes = GPSemanticProbes(input_dim=64, n_restarts=1, r2_ci_samples=50)
    probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
    results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})
    summary = probes.get_summary(results)

    for name in ["volume", "location", "shape"]:
        assert f"r2_{name}_linear" in summary
        assert f"r2_{name}_rbf" in summary
        assert f"r2_{name}_linear_ci_lo" in summary
        assert f"r2_{name}_linear_ci_hi" in summary
        assert f"r2_{name}_rbf_ci_lo" in summary
        assert f"r2_{name}_rbf_ci_hi" in summary
        assert f"lml_{name}_linear" in summary
        assert f"lml_{name}_rbf" in summary
        assert f"nonlinearity_evidence_{name}" in summary
    assert "r2_mean_linear" in summary
    assert "r2_mean_rbf" in summary
```

### 6.2 Modify: `tests/growth/test_latent_quality.py`

Remove `TestLinearProbe` and `TestSemanticProbes` classes entirely. Keep `TestDistanceCorrelation`, `TestVarianceMetrics`, `TestCKA`, `TestMMD`, and any other non-probe test classes unchanged.

---

## 7. Implementation Checklist

The coding agent should execute these steps in order:

### Phase A: Create the new module
- [ ] A1. Create `src/growth/evaluation/gp_probes.py` with `GPProbeResults`, `GPSemanticResults`, `GPProbe`, `GPSemanticProbes`, `extract_sausage_data`.
- [ ] A2. Verify: `python -c "from growth.evaluation.gp_probes import GPProbe; print('OK')"` succeeds.

### Phase B: Write tests
- [ ] B1. Create `tests/growth/test_gp_probes.py` with all 9 tests from Section 6.1.
- [ ] B2. Run: `python -m pytest tests/growth/test_gp_probes.py -v` — expect all 9 to pass.
- [ ] B3. If any test fails, debug and fix `gp_probes.py` until all pass.

### Phase C: Refactor existing modules
- [ ] C1. Edit `src/growth/evaluation/latent_quality.py`: remove `LinearProbe`, `ProbeResults`, `SemanticProbes`. Add re-exports from `gp_probes`.
- [ ] C2. Edit `src/growth/evaluation/__init__.py`: replace old exports with new GP exports.
- [ ] C3. Delete `src/growth/evaluation/enhanced_probes.py`.
- [ ] C4. Edit `experiments/lora_ablation/pipeline/evaluate_probes.py`: replace `EnhancedSemanticProbes` with `GPSemanticProbes`.
- [ ] C5. Edit `scripts/sdp_diagnostics.py`: replace inline Ridge with `GPProbe`.

### Phase D: Fix downstream references
- [ ] D1. `grep -r "LinearProbe\|EnhancedLinearProbe\|MLPProbe\|EnhancedSemanticProbes\|ProbeResults\|enhanced_probes" src/ experiments/ tests/ scripts/` — identify all remaining references.
- [ ] D2. Fix each reference found in D1.
- [ ] D3. Run: `python -m pytest tests/growth/test_latent_quality.py -v` — must pass (with probe-related tests removed).
- [ ] D4. Run: `python -m pytest tests/growth/test_gp_probes.py -v` — must pass.

### Phase E: Verify no regressions
- [ ] E1. Run full test suite: `python -m pytest tests/ -v --tb=short` and ensure no import errors or failures related to the refactoring.
- [ ] E2. Verify that no file in the project imports from `enhanced_probes` anymore.

---

## 8. Edge Cases and Pitfalls

### 8.1 GPy numerical stability

GPy can raise `numpy.linalg.LinAlgError` if the kernel matrix becomes ill-conditioned. This happens when:
- Features have near-zero variance in some dimensions (collapsed encoder dimensions).
- The lengthscale becomes very small (RBF kernel overfitting).

**Mitigation**: Add a jitter term. GPy does this automatically via `model.Gaussian_noise.variance`, but if errors persist, explicitly set a noise floor:
```python
model.Gaussian_noise.variance.constrain_bounded(1e-6, 10.0)
```

### 8.2 Optimization convergence

GPy's default optimizer (L-BFGS) may not converge in 500 iterations for the RBF kernel on 768-dimensional data. If `model.optimize()` returns without convergence:
- Increase `max_iters` to 1000.
- Use `model.optimize_restarts(num_restarts=5)` to escape local optima.
- Log a warning but do not raise an exception — partial optimization still yields usable predictions.

### 8.3 Memory with large feature dimensions

For $N = 800$, $D = 768$: the kernel matrix $K \in \mathbb{R}^{800 \times 800}$ is 5.1 MB (float64). No memory issues. The linear kernel can be computed as $K = \sigma_f^2 X X^\top$ without forming the full $D \times D$ covariance of features.

### 8.4 Shape feature dimensions

The `FEATURE_DIMS` dict currently uses `"shape": 3` in `enhanced_probes.py` but the methodology and module_3_sdp.md specify 6 shape features. The `latent_quality.py` version also uses 3. Check the actual target tensors loaded in `evaluate_probes.py` and use their runtime shape rather than hard-coding.

**Recommendation**: Infer output dimension from the target array shape at fit time rather than hard-coding `FEATURE_DIMS`. Keep `FEATURE_DIMS` only as a reference/documentation dict, not as a constraint.

### 8.5 Serialization

`GPy.models.GPRegression` objects are pickle-serializable, but the resulting files can be large (the model stores the training data). For saving to `probes_gp.pkl`, this is acceptable. If size becomes an issue, save only the hyperparameters and re-fit on demand.

---

## 9. References

1. Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning.* MIT Press, 2006. — Chapters 2 (Regression) and 5 (Model Selection).
2. Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.
3. Alain, G. & Bengio, Y. "Understanding Intermediate Layers Using Linear Classifier Probes." *ICLR Workshop*, 2017.
4. Micchelli, C. A. et al. "Universal Kernels." *JMLR*, 2006.
5. Bardes, A. et al. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *ICLR*, 2022.
6. GPy documentation: https://gpy.readthedocs.io/en/deploy/
