# Plan of Action: Meningioma Growth Prediction Framework

**Version:** 1.0  
**Date:** 2026-03-16  
**Author:** Mario Pascual González (with Claude — research assistant)  
**Repository:** MenGrowth-Model  

---

## 0. Executive Summary

This document specifies a three-stage experimental framework for meningioma growth prediction from longitudinal MRI, ordered by increasing complexity. Each stage must earn its place: Stage K+1 is only justified if it demonstrably outperforms Stage K under LOPO-CV. The stages are:

- **Stage 1 — Segmentation-Based Volumetric Baseline** (the "A0 ablation" elevated to a first-class citizen)
- **Stage 2 — Latent Severity Model** (your advisor's proposal, formalized as a nonlinear mixed-effects model)
- **Stage 3 — Representation-Learning Augmented Prediction** (the full BrainSegFounder → LoRA → SDP → GP pipeline)

A formal **variance decomposition protocol** quantifies the marginal contribution of each stage.

---

## 1. Stage 1 — Segmentation-Based Volumetric Baseline

### 1.1 Objective

Establish an empirical upper bound on what **volume-only** information can achieve. This is the baseline that every other approach must beat.

### 1.2 Pipeline: From MRI to Volume Trajectory

```
MenGrowth MRI [4, D, H, W]
    → BrainSegFounder (frozen encoder + adapted decoder)
    → Sliding-window inference (128³, overlap=0.5, Gaussian weighting)
    → Argmax → Binary WT mask (labels 1∪2∪3)
    → Volume = Σ(mask > 0) × voxel_volume_mm³
    → y = log(V + 1)   [log1p transform]
    → Per-patient trajectory: {(t_k, y_k)}_{k=1}^{n_i}
```

**Volume extraction** is already implemented in `src/growth/data/semantic_features.py::compute_volumes()`. The function counts voxels per BraTS label and multiplies by voxel volume (spacing is 1mm³ isotropic after BraTS preprocessing, so `voxel_volume = 1.0`). The `compute_log_volumes()` function applies `log1p`. This infrastructure is sound and tested (`tests/growth/test_semantic_features.py`).

**Segmentation models** are configured in `experiments/segment_based_approach/config.yaml`. Currently three are enabled: `brainsegfounder` (frozen original), `bsf_adapted_decoder_men_domain` (decoder fine-tuned on BraTS-MEN), and `bsf_lora_r8_adapted_men_domain` (LoRA r=8 + adapted decoder). The run script `experiments/segment_based_approach/run_baseline.py` orchestrates extraction and LOPO-CV.

### 1.3 Growth Models (Three-Model Hierarchy)

All three models already exist in `src/growth/models/growth/`. The following specifies how each should be configured for the volumetric baseline.

#### 1.3.1 ScalarGP (Model A)

**File:** `src/growth/models/growth/scalar_gp.py`  
**Library:** GPy  
**Configuration:**
- Input: `t ∈ ℝ` (ordinal timepoint, or days-since-baseline when timestamps arrive)
- Output: `y = log(V + 1) ∈ ℝ`
- Kernel: `Matérn-5/2(t; σ²_f, ℓ) + WhiteNoise(σ²_n)`
- Mean function: `linear` (constant + slope)
- Optimization: L-BFGS-B, 5 restarts, 1000 max iterations
- Bounds: lengthscale ∈ [0.1, 50.0], signal variance ∈ [0.001, 10.0], noise variance ∈ [1e-6, 5.0]

**Mathematical formulation:**

$$y_i(t) \sim \mathcal{GP}\bigl(\beta_0 + \beta_1 t,\; k_{\text{Mat52}}(t, t') + \sigma^2_n \delta(t, t')\bigr)$$

where:

$$k_{\text{Mat52}}(r) = \sigma^2_f \left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5\,r^2}{3\,\ell^2}\right) \exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right), \quad r = |t - t'|$$

This model pools all patients into a single GP. It has **5 hyperparameters** (β₀, β₁, σ²_f, ℓ, σ²_n) — within budget for N=31.

#### 1.3.2 LME (Model B)

**File:** `src/growth/models/growth/lme_model.py`  
**Library:** `statsmodels` (REML)  
**Configuration:**
- Fixed effects: intercept + slope on time
- Random effects: per-patient intercept + slope
- Prediction via BLUP (Best Linear Unbiased Prediction)

**Mathematical formulation:**

$$y_{ij} = (\beta_0 + u_{0i}) + (\beta_1 + u_{1i})\,t_{ij} + \varepsilon_{ij}$$

where $u_i = (u_{0i}, u_{1i})^T \sim \mathcal{N}(0, \Omega)$ and $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$.

**Parameter count:** 2 fixed effects + 3 variance components (var(u₀), var(u₁), cov(u₀,u₁)) + 1 residual variance = **6 parameters** estimated from the full dataset. Per-patient random effects are BLUPs, not free parameters.

#### 1.3.3 Hierarchical GP (Model C)

**File:** `src/growth/models/growth/hgp_model.py`  
**Library:** GPy  
**Configuration:**
- Population mean: linear, derived from LME fit (β̂₀, β̂₁)
- Kernel: Matérn-5/2 on residuals (observations − population mean)
- Hyperparameters pooled across all patients

**Mathematical formulation:**

$$y_i(t) = m(t) + f_i(t) + \varepsilon_i(t)$$

where $m(t) = \hat{\beta}_0 + \hat{\beta}_1 t$ (from LME), $f_i(t) \sim \mathcal{GP}(0, k_{\text{Mat52}})$ captures individual deviations, and hyperparameters (σ²_f, ℓ, σ²_n) are shared across patients via empirical Bayes (marginal likelihood optimization on pooled residuals).

### 1.4 Covariates to Add When Available

When timestamps, age, and sex become available, incorporate them as **static covariates** in the GP mean function (not the kernel, to avoid curse of dimensionality):

$$m(t; \mathbf{x}_i) = \beta_0 + \beta_1 t + \beta_2 \cdot \text{age}_i + \beta_3 \cdot \text{sex}_i$$

The `covariate_utils.py` module already supports this via `get_patient_covariate_vector()`. The config key `covariates.enabled: true` activates it.

**Parameter impact:** Adding age and sex costs +2 mean-function parameters. At N=31 this is tight; at N=58 it is acceptable.

### 1.5 What Else Can Be Done for This Baseline?

**Currently missing improvements:**

1. **Gompertz mean function.** The current implementation offers `linear` and `zero` mean functions (config `hgp.mean_function`). Literature (Engelhardt et al., 2023; Vaghi et al., 2020) establishes Gompertz as the best parametric model for meningioma growth. A Gompertz mean function would be:

$$m(t) = V_{\max} \exp\!\bigl(-\exp(-\alpha(t - t_{\text{mid}}))\bigr)$$

**Implementation:** Add a `gompertz` option to `hgp_model.py` using `GPy.mappings.Kernel` or a custom `GPy.core.Mapping` subclass. This adds 3 parameters (V_max, α, t_mid) — borderline at N=31, but the GP regularization via the kernel prior mitigates overfitting.

**Package:** GPy supports custom mean functions via `GPy.core.GP(mean_function=...)`. Alternatively, fit Gompertz parametrically first (via `scipy.optimize.curve_fit`), then use the fitted curve as a fixed mean for the GP — this costs 0 additional GP parameters.

2. **Bootstrap confidence intervals on LOPO-CV metrics.** The current `LOPOEvaluator` reports point estimates. Add `.632+ bootstrap` (Efron & Tibshirani, 1997) for CI estimation.

**Implementation:** After LOPO-CV, resample the per-patient error vector with replacement (B=2000 iterations), compute the metric on each resample, report 2.5th and 97.5th percentiles. Use `scipy.stats.bootstrap` (available since scipy 1.9) or `sklearn.utils.resample`.

3. **Scanner effect testing.** Before applying ComBat, test whether scanner effects are statistically significant for volumetric features. A Kruskal-Wallis test across scanner groups on log-volume residuals (after detrending for time) is sufficient. If p > 0.05, ComBat is unnecessary for volume and may introduce artifacts.

**Package:** `scipy.stats.kruskal` for the test. `neuroCombat` (Python) for harmonization if needed.

### 1.6 Testable Outcomes (Stage 1)

| Test ID | Description | Pass Criterion |
|---------|-------------|----------------|
| S1-T1 | ScalarGP LOPO-CV completes without NaN | All 30 folds produce finite predictions |
| S1-T2 | LME R²_log > 0 | LME captures temporal trend (current: 0.387) |
| S1-T3 | HGP R²_log ≥ ScalarGP R²_log | Hierarchical structure does not degrade |
| S1-T4 | Calibration_95 ∈ [0.85, 1.0] | Prediction intervals are well-calibrated |
| S1-T5 | Bootstrap 95% CI on R²_log excludes 0 for best model | Prediction is statistically significant |
| S1-T6 | Per-patient error distribution is reported | Full distribution, not just mean |
| S1-T7 | Gompertz mean function tested as ablation | Compare linear vs Gompertz mean in HGP |

---

## 2. Stage 2 — Latent Severity Model

### 2.1 Objective

Implement the advisor's proposal: a single latent variable $s_i \in [0, 1]$ ("severity") that governs each patient's growth trajectory, with monotonicity constraints in both time and severity.

### 2.2 Mathematical Formalization

The model is a **nonlinear mixed-effects model** (NLME) with a single random effect:

$$q_{ij} = g(s_i, t_{ij};\, \boldsymbol{\theta}) + \varepsilon_{ij}, \quad \varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$$

where:

- $q_{ij} \in [0, 1]$ is the **growth quantile** for patient $i$ at time $j$
- $s_i \in [0, 1]$ is the **latent severity** (constant per patient)
- $t_{ij} \in [0, 1]$ is the **normalized time** (quantile of elapsed time from baseline)
- $\boldsymbol{\theta}$ are shared population parameters
- $g$ satisfies: (i) $g(s, 0) = 0\;\forall s$, (ii) $\partial g / \partial t \geq 0$, (iii) $\partial g / \partial s \geq 0$

### 2.3 How to Compute q and t (The Quantile Transform)

This is a critical implementation detail from the PDF proposal:

**Time quantile $t_{ij}$:** Let $\Delta t_{ij}$ be the elapsed time from baseline for patient $i$ at observation $j$ (in days, when timestamps arrive; currently ordinal indices). Compute the **empirical CDF across all patients' time intervals**:

$$t_{ij} = \hat{F}_{\Delta t}\bigl(\Delta t_{ij}\bigr) = \frac{\text{rank}(\Delta t_{ij})}{N_{\text{total}} + 1}$$

where $N_{\text{total}}$ is the total number of temporal observations across all patients. This maps heterogeneous follow-up intervals to [0, 1].

**Growth quantile $q_{ij}$:** Let $V_{ij}$ be the volume at timepoint $j$ and $V_{i0}$ the baseline volume. The raw growth is $\Delta V_{ij} = V_{ij} - V_{i0}$ (or the log-ratio: $\log(V_{ij}/V_{i0})$). Then:

$$q_{ij} = \hat{F}_{\Delta V}\bigl(\Delta V_{ij}\bigr)$$

the empirical quantile of growth across all observations. This naturally satisfies $q_{i0} = 0$ if we define $\Delta V_{i0} = 0$ and use a convention where the zero-growth quantile maps to 0.

**Warning:** The quantile transform is a **rank-based nonlinear transformation**. It destroys magnitude information — a patient growing by 1 cm³ and another by 100 cm³ may be adjacent in quantile space if most patients grow between 10–90 cm³. This is a design choice that prioritizes ordinal correctness over absolute accuracy. The advisor's proposal explicitly operates in quantile space, so this is intentional.

**Implementation:** Use `scipy.stats.rankdata` with `method='average'`, divide by `(n + 1)`. Apply this at the beginning of the pipeline, before model fitting.

### 2.4 How to Compute the Severity Variable $s_i$

This is the core algorithmic challenge. The severity is **latent** — it is not directly observed. There are three approaches, in order of recommendation:

#### Approach A: Joint MLE / Bayesian Estimation (Recommended)

Treat $\{s_i\}$ and $\boldsymbol{\theta}$ as unknowns to be jointly estimated from the training data. For each patient, $s_i$ is a free parameter constrained to $[0, 1]$.

**Optimization problem:**

$$\min_{\boldsymbol{\theta},\, \{s_i\}} \sum_{i=1}^{N} \sum_{j=1}^{n_i} \bigl(q_{ij} - g(s_i, t_{ij};\, \boldsymbol{\theta})\bigr)^2 + \lambda \|\boldsymbol{\theta}\|^2$$

subject to $s_i \in [0, 1]$, $\boldsymbol{\theta}$ ensuring monotonicity.

**Parameter count:** $|\boldsymbol{\theta}|$ (model weights) + $N$ (one severity per patient). For N=31 patients and a 2-weight logistic model: 2 + 31 = 33 parameters from ~112 observations. This is tight but feasible because the $s_i$ are individually well-constrained by each patient's trajectory (2–5 observations per patient constraining a single scalar).

**Package:** `scipy.optimize.minimize` with `method='L-BFGS-B'` and `bounds=[(0, 1)] * N + [(...)]` for θ. Alternatively, **PyTorch** with `torch.sigmoid` to enforce [0,1] on $s_i$ and positive-weight constraints on $\boldsymbol{\theta}$.

#### Approach B: Bayesian via Stan/brms (R, if warranted)

For principled posterior uncertainty on severity:

```r
# R code using brms (Bayesian NLME via Stan)
library(brms)

# Custom family for constrained growth model
growth_formula <- bf(
  q ~ inv_logit(w1 * s + w2 * t) * t,  # Ensures q(s,0)=0
  s ~ 1 + baseline_vol + age,            # Severity predicted from covariates
  w1 + w2 ~ 1,                           # Population-level weights
  nl = TRUE
)
```

**Package:** `brms` (R), which compiles to Stan. Python alternative: `pymc` (v5) or `numpyro`.

**When to use R:** If posterior uncertainty on severity is important for the thesis (e.g., for credible intervals on patient-level growth trajectories). The `brms` package handles NLME with latent variables more naturally than any Python equivalent. The R script can be called from the Python pipeline via `subprocess` or `rpy2`.

#### Approach C: Two-Step (Simpler but Suboptimal)

1. Fit the growth model with known-severity proxies (e.g., use baseline volume percentile as an initial $\hat{s}_i$)
2. Refine $s_i$ by minimizing prediction error per patient

This is computationally simpler but introduces a chicken-and-egg circularity. Only use as initialization for Approach A.

### 2.5 The Growth Function $g(s, t; \boldsymbol{\theta})$

The advisor suggests constrained logistic regression. Here are three concrete options:

#### Option 1: Weighted Sigmoid with Boundary Constraint

$$g(s, t; w_1, w_2, b) = t \cdot \sigma(w_1 \cdot s + w_2 \cdot t + b)$$

where $\sigma$ is the logistic sigmoid and $w_1, w_2 > 0$ are constrained positive. The multiplication by $t$ ensures $g(s, 0) = 0$ exactly. Monotonicity in $t$ is approximately (but not exactly) guaranteed — the derivative $\partial g / \partial t = \sigma(\cdot) + t \cdot \sigma'(\cdot) \cdot w_2$ is positive when $w_2 > 0$ and $t$ is not too large.

**Parameters:** 3 (w₁, w₂, b) + N severity values

#### Option 2: Constrained Monotonic Neural Network (CMNN)

Use the `MonoDense` layer from Runje & Shankaranarayana (2023):

```python
# Using the monotonic-nn library (pip install monotonic-nn)
from monotonicnn import MonoDense

# 2 inputs: [s, t], both monotonically increasing
model = nn.Sequential(
    MonoDense(2, 16, monotonicity_indicator=[1, 1]),  # Both inputs monotonic increasing
    nn.ReLU(),
    MonoDense(16, 1, monotonicity_indicator=[1])
)
# Multiply output by t to enforce g(s, 0) = 0
```

**Package:** `pip install monotonic-nn` (PyTorch). Alternatively, implement weight clipping manually: `weight.data.clamp_(min=0)` after each optimizer step.

**Parameters:** ~50 for a small 2-layer network. At N=31, this is **over-parameterized** — regularization is critical.

#### Option 3: Reduced Gompertz with Severity (Recommended)

Following Vaghi et al. (2020), use the reduced Gompertz model where the growth rate parameter is a function of severity:

$$V(t) = V_0 \exp\!\left(\frac{\alpha(s_i)}{\beta}\bigl(1 - e^{-\beta t}\bigr)\right)$$

where $\alpha(s_i) = \alpha_0 + \alpha_1 \cdot s_i$ (growth rate increases with severity) and $\beta$ is the deceleration rate (shared across patients). Then:

$$q_{ij} = \hat{F}\!\left(\log\frac{V(t_{ij})}{V_0}\right)$$

**Parameters:** 3 population parameters (α₀, α₁, β) + N severity values. This has the strongest biological grounding (Gompertz dynamics) and the fewest free population parameters.

**Package:** `scipy.optimize.curve_fit` for initial MLE, then refine with PyTorch/scipy for joint (θ, s) optimization.

### 2.6 Test-Time Severity Estimation

At test time, given a **single baseline MRI**, we need to estimate $\hat{s}_{\text{new}}$.

**Approach: Severity Regression Head**

Train a lightweight regressor mapping baseline features to severity:

$$\hat{s}_{\text{new}} = \sigma\!\bigl(\mathbf{w}^T \mathbf{x}_{\text{baseline}} + b\bigr)$$

where $\mathbf{x}_{\text{baseline}}$ includes:

- $\log(V_{\text{baseline}} + 1)$: baseline tumor volume (from segmentation)
- Age at baseline (when available)
- Sex (when available)
- Sphericity (from `compute_sphericity()` in `semantic_features.py`)

This is a **3–4 feature logistic regression** — well within the budget for N=31. Train it on the training patients' estimated severities $\{\hat{s}_i\}$ from the joint optimization.

**For the representation-learning pathway (Stage 3):** Replace hand-crafted features with BrainSegFounder encoder features (768-dim → PCA → 3–5 dims → severity regression). This is how Stage 2 connects to Stage 3.

### 2.7 Integration with Existing Codebase

**New files to create:**

```
experiments/severity_model/
├── config.yaml
├── run_severity.py          # Orchestrator
├── __init__.py
src/growth/models/growth/
├── severity_model.py         # Core NLME with severity
├── quantile_transform.py     # q and t quantile computation
```

**Modifications to existing files:**
- `src/growth/models/growth/__init__.py`: Add `SeverityModel` to exports
- `src/growth/evaluation/lopo_evaluator.py`: Support models that jointly optimize severity (need to re-estimate $\hat{s}_{\text{test}}$ at each LOPO fold from the severity regression head, **not** from the joint optimization — that would be data leakage)

### 2.8 Testable Outcomes (Stage 2)

| Test ID | Description | Pass Criterion |
|---------|-------------|----------------|
| S2-T1 | Quantile transform produces values in [0, 1] | All q, t ∈ [0, 1] |
| S2-T2 | Joint optimization converges | Loss decreases monotonically, final loss < initial |
| S2-T3 | Monotonicity satisfied | $\partial g/\partial s \geq 0$ and $\partial g/\partial t \geq 0$ at 100 test points |
| S2-T4 | Boundary condition holds | $g(s, 0) = 0$ exactly for all s |
| S2-T5 | Severity values spread across [0, 1] | std(s) > 0.15 (not collapsed to a point) |
| S2-T6 | Severity correlates with known clinical proxies | Spearman(s, baseline_volume) > 0.3 |
| S2-T7 | LOPO-CV R²_log ≥ Stage 1 best | Severity model matches or exceeds volumetric baseline |
| S2-T8 | Severity regression head from baseline features | LOO R² > 0 for severity prediction |

---

## 3. Stage 3 — Representation-Learning Augmented Prediction

### 3.1 Objective

Test whether BrainSegFounder latent representations capture growth-relevant information **beyond volume**, justifying the added pipeline complexity.

### 3.2 Full Pipeline Specification

```
BraTS-MEN MRI [4, 128³]
  → BrainSegFounder Encoder (frozen or LoRA-adapted)
  → GAP → h ∈ ℝ^768
  → SDP Network → z ∈ ℝ^128 = [z_vol(32) | z_residual(96)]
  → PCA on z_residual → z̃_residual ∈ ℝ^k  (k chosen by 90% variance)
  → Concatenate: [log(V+1), z̃_residual_1, ..., z̃_residual_k]
  → GP/LME with ARD kernel
  → Growth prediction ± uncertainty
```

### 3.3 LoRA Selection Protocol

**Current setup:** LoRA ranks {2, 4, 8, 16, 32} are trained. Selection should be based on **downstream utility for growth prediction**, not segmentation Dice.

**Selection metric hierarchy:**

1. **GP probe R² on held-out volume prediction** (primary)
2. **VICReg loss on validation set** (secondary — representation quality)
3. **Segmentation Dice** (tertiary — sanity check only)

The existing probe infrastructure (`src/growth/evaluation/`) computes GP probes with linear and RBF kernels. Run probes for each LoRA rank and select the one with highest R² on volume prediction.

**Recommendation:** Start with LoRA r=4 or r=8. At N=31, higher ranks (16, 32) risk overfitting the adapter to BraTS-MEN segmentation without improving downstream representation quality.

### 3.4 SDP Module: What the Residual Partition Should Encode

The SDP module (768 → 128 dims) currently uses vol_dim=32 and residual_dim=96. After R1, the volume partition is supervised with $\lambda_{\text{vol}} = 25.0$ and the residual partition is regularized only via VICReg (covariance, variance) and distance correlation from the volume partition.

**The residual partition should encode texture, heterogeneity, and morphological features that are not captured by volume alone.** The VICReg loss encourages the residual dimensions to be informative (variance hinge) and decorrelated from volume (cross-partition dCor). However, without explicit supervision, the residual may encode scanner artifacts, noise, or volume-correlated redundancy.

**Proposed improvement — Supervised residual targets:**

Add **weak supervision** on a subset of residual dimensions using features that are (a) computable from the segmentation mask and (b) not highly correlated with volume:

- **Sphericity** (already in `compute_sphericity()`): Measures shape compactness, r ≈ 0.3 with volume
- **Enhancement ratio** (already in `compute_composition_features()`): $V_{\text{ET}} / V_{\text{total}}$
- **Infiltration index**: $V_{\text{ED}} / (V_{\text{NCR}} + V_{\text{ET}})$

Assign 8 residual dimensions to these 3 targets (e.g., residual_supervised_dim=8, residual_free_dim=88). This gives the residual partition a fighting chance to learn beyond-volume features while keeping 88 dimensions for unsupervised structure.

**Modification:** In `src/growth/config/phase2_sdp.yaml`, add:

```yaml
partition:
  vol_dim: 32
  residual_supervised_dim: 8   # NEW: supervised texture/shape
  residual_free_dim: 88        # Unsupervised via VICReg only
targets:
  n_vol: 1
  n_residual_supervised: 3     # sphericity, enhancement_ratio, infiltration_index
```

### 3.5 PCA Compression: From 128 to k Dimensions

**Why PCA on residuals, not on the full z:** The volume partition (32 dims) is already well-supervised and low-dimensional enough. The problem is the 96-dim residual partition, which at N=31 creates a 96:31 parameter-to-patient disaster for any GP.

**Protocol:**

1. Extract z_residual ∈ ℝ^96 for all N patients × all timepoints (N_obs ≈ 112)
2. Fit PCA on training patients' residuals (exclude test patient in LOPO)
3. Select k components retaining ≥ 90% variance (expect k ≈ 5–15)
4. Transform: $\tilde{z}_{\text{res}} = \text{PCA}_k(z_{\text{residual}})$
5. Concatenate with scalar volume: input to GP = $[\log(V+1),\, \tilde{z}_{\text{res}_1},\, \ldots,\, \tilde{z}_{\text{res}_k}]$

**Package:** `sklearn.decomposition.PCA` with `n_components=0.9` (variance threshold).

**Critical detail:** PCA must be fit **inside each LOPO fold** on the training patients only. Fitting on the full dataset and then doing LOPO is data leakage.

### 3.6 GP with ARD Kernel on Reduced Features

The GP kernel for multi-dimensional input becomes:

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2_f \prod_{d=1}^{D} k_{\text{Mat52}}\!\left(\frac{|x_d - x'_d|}{\ell_d}\right) + \sigma^2_n \delta(\mathbf{x}, \mathbf{x}')$$

where ARD (Automatic Relevance Determination) assigns a per-dimension lengthscale $\ell_d$. Dimensions that are irrelevant for prediction get large $\ell_d$ (effectively ignored). This is soft feature selection built into the GP.

**Parameter count:** D lengthscales + 1 signal variance + 1 noise variance + mean function parameters. For D = 1 (time) + 1 (volume) + k (PCA residuals), with k=5: total ≈ 10 kernel hyperparameters. At N=31 × ~3.6 obs/patient ≈ 112 observations, this is feasible but requires careful regularization (informative priors on lengthscales).

**Package:** GPy with `GPy.kern.Matern52(input_dim=D, ARD=True)`.

### 3.7 Integration with Severity Model (Stage 2 × Stage 3)

The killer combination: use **BrainSegFounder features to estimate severity** instead of hand-crafted features.

```
Baseline MRI → BrainSegFounder → GAP → h ∈ ℝ^768
  → PCA → h̃ ∈ ℝ^10
  → Severity regression: ŝ = σ(w^T h̃ + b)
  → Feed ŝ into Stage 2 severity growth model
```

This provides a principled test: do deep features improve severity estimation beyond what [volume, age, sex, sphericity] provide?

**Testable comparison:**

| Severity Estimation Input | Features | # Params |
|--------------------------|----------|----------|
| Clinical only | [log_vol, age, sex] | 4 |
| Clinical + shape | [log_vol, age, sex, sphericity] | 5 |
| Deep features (PCA) | [PCA_1, ..., PCA_5] | 6 |
| Clinical + deep | [log_vol, age, sex, PCA_1, ..., PCA_3] | 7 |

Compare via cross-validated R² of severity prediction.

### 3.8 Testable Outcomes (Stage 3)

| Test ID | Description | Pass Criterion |
|---------|-------------|----------------|
| S3-T1 | GP probe R² per LoRA rank computed | Best rank identified |
| S3-T2 | SDP quality report: dCor(vol, residual) < 0.15 | Partitions are decorrelated |
| S3-T3 | PCA on residual: k components for 90% variance | k reported (expect 5–15) |
| S3-T4 | LOPO-CV with [volume + PCA_residual] ≥ Stage 1 | Deep features add signal |
| S3-T5 | Severity estimation: deep features vs clinical | R² comparison reported |
| S3-T6 | ARD lengthscales: which residual dims matter? | Report per-dim relevance |

---

## 4. Variance Decomposition Protocol

### 4.1 Purpose

Formally quantify how much predictive variance each component explains. This is the **central analytical contribution** of the thesis — it answers "was the complexity worth it?"

### 4.2 Mathematical Framework

Define the following models in order of complexity:

| Model | Input | # Effective Params | Notation |
|-------|-------|-------------------|----------|
| M₀ | Population mean only | 2 | $\hat{y}^{(0)}$ |
| M₁ | Volume + time (ScalarGP) | 5 | $\hat{y}^{(1)}$ |
| M₂ | Volume + time (LME/HGP) | 6–8 | $\hat{y}^{(2)}$ |
| M₃ | Volume + time + severity | 3 + N | $\hat{y}^{(3)}$ |
| M₄ | Volume + time + severity + deep features | 3 + N + k | $\hat{y}^{(4)}$ |

For each model $M_k$, compute LOPO-CV:

$$R^2_k = 1 - \frac{\sum_{i} (y_i - \hat{y}^{(k)}_i)^2}{\sum_{i} (y_i - \bar{y})^2}$$

The **marginal contribution** of component $k$ is:

$$\Delta R^2_k = R^2_k - R^2_{k-1}$$

### 4.3 Statistical Significance of Marginal Contributions

Use a **paired permutation test** on per-patient LOPO errors:

$$H_0: \text{MAE}(M_k) = \text{MAE}(M_{k-1})$$

For each of B=10000 permutations, randomly swap each patient's error between the two models, compute the difference in mean MAE, and check if the observed difference falls in the extreme tail.

**Package:** `scipy.stats.permutation_test` (scipy ≥ 1.9) or manual implementation.

### 4.4 Reporting Format

For the thesis, produce a table like:

```
| Model           | R²_log | ΔR²   | p-value | MAE_log | Cal_95 |
|-----------------|--------|-------|---------|---------|--------|
| M₀: Mean only   | ...    | —     | —       | ...     | ...    |
| M₁: ScalarGP    | ...    | ...   | ...     | ...     | ...    |
| M₂: HGP         | ...    | ...   | ...     | ...     | ...    |
| M₃: Severity    | ...    | ...   | ...     | ...     | ...    |
| M₄: Deep+Sev    | ...    | ...   | ...     | ...     | ...    |
```

Plus a **bar chart** of ΔR² with bootstrap 95% CIs, and a **per-patient error scatter plot** showing which patients improve/degrade with each model.

### 4.5 Implementation

**New file:** `experiments/variance_decomposition/run_decomposition.py`

This orchestrator:
1. Runs each model under identical LOPO-CV folds
2. Collects per-patient predictions and errors
3. Computes ΔR², paired permutation tests, bootstrap CIs
4. Generates summary table and figures

**Package dependencies:**
- `scipy.stats.permutation_test`
- `matplotlib` for figures
- Existing `LOPOEvaluator` for consistent fold management

---

## 5. Package and Library Reference

### 5.1 Python (Primary)

| Purpose | Package | Version | Notes |
|---------|---------|---------|-------|
| GP models | `GPy` | ≥1.10.0 | ScalarGP, HGP (already in use) |
| LME models | `statsmodels` | ≥0.14.0 | REML, BLUP (already in use) |
| Optimization | `scipy.optimize` | ≥1.9 | L-BFGS-B for severity model |
| PCA | `sklearn.decomposition.PCA` | ≥1.3 | Variance-threshold selection |
| Bootstrap | `scipy.stats.bootstrap` | ≥1.9 | CI estimation |
| Permutation test | `scipy.stats.permutation_test` | ≥1.9 | ΔR² significance |
| Quantile transform | `scipy.stats.rankdata` | ≥1.7 | For q and t computation |
| Monotonic NN | `monotonic-nn` | ≥1.0 | CMNN layers (if Option 2 in §2.5) |
| ComBat | `neuroCombat` | ≥0.2 | Multi-scanner harmonization |
| Deep learning | `PyTorch` | ≥2.0 | LoRA, SDP, encoder (already in use) |
| Config | `omegaconf` | ≥2.3 | YAML config (already in use) |
| HDF5 | `h5py` | ≥3.8 | Data I/O (already in use) |

### 5.2 R (If Warranted by Specific Capabilities)

| Purpose | Package | When to Use |
|---------|---------|-------------|
| Bayesian NLME | `brms` (→ Stan) | If posterior uncertainty on severity is needed for thesis |
| Monotonic GP | `lineqGPR` | If monotonic GP is preferred over CMNN |
| Sample size calculation | `pmsampsize` | For formal reporting of statistical adequacy |

**Integration with Python:** Use `subprocess.run(["Rscript", "path/to/script.R"])` for one-off analyses. For tighter integration, use `rpy2`. Place R scripts in `scripts/r/` with clear documentation.

---

## 6. Repository Integration Plan

### 6.1 New Directory Structure

```
experiments/
├── segment_based_approach/     # Stage 1 (EXISTS)
│   ├── config.yaml
│   ├── run_baseline.py
│   └── segment.py
├── severity_model/             # Stage 2 (NEW)
│   ├── config.yaml
│   ├── run_severity.py
│   ├── __init__.py
│   └── README.md
├── deep_features/              # Stage 3 (NEW)
│   ├── config.yaml
│   ├── run_deep_prediction.py
│   ├── __init__.py
│   └── README.md
├── variance_decomposition/     # Cross-stage analysis (NEW)
│   ├── config.yaml
│   ├── run_decomposition.py
│   ├── __init__.py
│   └── README.md
└── sdp/                        # SDP training (EXISTS)

src/growth/models/growth/
├── __init__.py                 # UPDATE: add SeverityModel
├── base.py                     # EXISTS
├── scalar_gp.py                # EXISTS (Stage 1)
├── lme_model.py                # EXISTS (Stage 1)
├── hgp_model.py                # EXISTS (Stage 1)
├── severity_model.py           # NEW (Stage 2)
├── quantile_transform.py       # NEW (Stage 2)
└── covariate_utils.py          # EXISTS

src/growth/evaluation/
├── lopo_evaluator.py           # UPDATE: support severity re-estimation per fold
├── variance_decomposition.py   # NEW: ΔR², permutation tests, bootstrap CIs
└── ...
```

### 6.2 Dependency Order

```
Stage 1 (Segment-Based Baseline)
    ↓ establishes R²_baseline
Stage 2 (Severity Model)
    ↓ requires: quantile_transform.py, severity_model.py
    ↓ tests: R²_severity > R²_baseline ?
Stage 3 (Deep Features)
    ↓ requires: LoRA selection, SDP residual supervision, PCA pipeline
    ↓ tests: R²_deep > R²_severity ?
Variance Decomposition
    ↓ requires: all stages completed
    ↓ produces: final comparison table
```

### 6.3 Configuration Inheritance

All experiment configs should inherit shared settings from `src/growth/config/phase4_growth.yaml`:

```yaml
# experiments/severity_model/config.yaml
defaults:
  - /growth/config/phase4_growth

severity:
  growth_function: gompertz_reduced  # option1 | option2_cmnn | gompertz_reduced
  optimization:
    method: lbfgsb
    max_iter: 5000
    n_restarts: 10
    lambda_reg: 0.01
  severity_estimation:
    features: [log_volume, age, sex, sphericity]
    model: logistic_regression
```

---

## 7. Priority Ordering and Timeline

### 7.1 Critical Path

```
[PRIORITY 1] Obtain timestamps + age/sex metadata
             ↓ (Unblocks proper temporal modeling)
[PRIORITY 2] Stage 1: Finalize volumetric baseline with Gompertz mean + bootstrap CIs
             ↓ (Establishes R²_baseline with confidence intervals)
[PRIORITY 3] Stage 2: Implement severity model (quantile transform + joint optimization)
             ↓ (Tests advisor's proposal)
[PRIORITY 4] Dataset expansion to N=54-58
             ↓ (Crosses statistical adequacy thresholds)
[PRIORITY 5] Stage 3: Deep feature augmentation (LoRA selection + SDP residual + PCA)
             ↓ (Tests representation learning value-add)
[PRIORITY 6] Variance decomposition + thesis writing
```

### 7.2 What to Implement Now (N=31, Ordinal Time)

Even without timestamps, the following is immediately actionable:

1. **Stage 1:** Run the existing `experiments/segment_based_approach/run_baseline.py` to completion. Verify the reported numbers (ScalarGP R²=-0.12, LME R²=0.39, HGP R²=-0.04). Add bootstrap CIs.

2. **Stage 2:** Implement `quantile_transform.py` using ordinal indices as time. The quantile transform works regardless of whether time is in days or ordinal. Implement `severity_model.py` with the reduced Gompertz option. Run LOPO-CV.

3. **Variance decomposition:** Compare M₁ vs M₂ vs M₃.

### 7.3 What Requires Timestamps

- Proper Matérn-5/2 temporal correlation (currently, ordinal indices distort inter-observation gaps)
- Gompertz mean function calibration (growth rate α has units of 1/time)
- Clinical interpretability of predictions ("tumor will grow by X% in Y months")

### 7.4 What Requires N=54-58

- Stage 3 with >5 input dimensions to the GP (currently risky at N=31)
- Adding age + sex as covariates simultaneously
- Reliable comparison of all 5 models in the variance decomposition

---

## 8. Open Questions and Decisions

1. **Quantile space vs. log-volume space:** The severity model (Stage 2) operates in quantile space per the advisor's proposal. Stages 1 and 3 operate in log-volume space. The variance decomposition must compare models in the **same output space**. Decision: report both, with log-volume as primary (clinically interpretable) and quantile R² as secondary.

2. **Gompertz mean in HGP vs. Gompertz in severity model:** These are related but distinct. The HGP Gompertz mean is a fixed parametric trend; the severity model makes the growth rate patient-specific. If severity works well, the HGP Gompertz mean becomes redundant (severity already explains inter-patient variation).

3. **SDP residual supervision budget:** Adding 3 supervised features to the residual partition costs 3 regression heads + 8 latent dimensions. At BraTS-MEN N≈800, this is fine. But it means the SDP module needs retraining — plan for this before Stage 3.

4. **R vs. Python for severity model:** Start in Python (scipy + PyTorch). Switch to R/brms only if (a) posterior credible intervals on severity are needed for the thesis, or (b) Python optimization fails to converge reliably.
