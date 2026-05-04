# Stage 1 — Uncertainty-Propagated Volume Prediction

**Spec version:** v1.0 (2026-04-28)
**Author:** Mario Pascual González (in collaboration with Claude)
**Target:** local Claude Code agent operating on
`/home/mpascual/research/code/MenGrowth-Model/`
**Conda env:** `~/.conda/envs/growth/bin/python`
**Test command:** `~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short`

---

## 0. TL;DR for the agent

Build a **heteroscedastic-residual** version of the existing volumetric growth
pipeline. Inputs come from the `uncertainty/` group of `MenGrowth.h5` (already
populated by the LoRA-Ensemble run, rank=32, M=20). Two new growth models
(`LMEHetero`, `ScalarGPHetero`) plus an `HGPHetero` ablation are required. The
existing homoscedastic models (`LMEGrowthModel`, `ScalarGP`,
`HierarchicalGPModel`) are kept as comparators, so that the **value of the
propagated uncertainty** can be quantified via paired ΔR², CRPS, and calibration
metrics. A new orchestrator `experiments/stage1_volumetric/run_stage1_uq.py`
runs the full LOPO-CV, emits paired comparisons, and accepts a `--real-time`
flag to switch from ordinal time to days-from-baseline derived from
`metadata/study_date`.

The PatientTrajectory dataclass is extended with an optional
`observation_variance` field; existing code paths (which never set it) remain
fully backwards-compatible.

You **must read this document in full before writing code**. The math in §2 and
§3 dictates the exact form of the likelihood and the prediction equations, and
silently substituting a different formulation (for example, using
`statsmodels.MixedLM` weights) would silently break the propagation.

---

## 1. Project context (read these files before anything else)

**Existing infrastructure to reuse — do not duplicate**

| Path | Why |
|------|-----|
| `src/growth/shared/growth_models.py` | `GrowthModel` ABC, `PatientTrajectory`, `FitResult`, `PredictionResult`. Extend `PatientTrajectory` here. |
| `src/growth/shared/lopo.py` | `LOPOEvaluator` — reuse as-is for the new models. |
| `src/growth/shared/metrics.py` | Add CRPS and multi-level coverage. |
| `src/growth/shared/bootstrap.py` | Reuse `bootstrap_metric`, `paired_permutation_test`. |
| `src/growth/shared/covariate_utils.py` | Reuse `collect_covariates`, `get_patient_covariate_vector`. |
| `src/growth/models/growth/lme_model.py` | Reference homoscedastic LME — mimic its public API. |
| `src/growth/models/growth/scalar_gp.py` | Reference homoscedastic GP — mimic its public API. |
| `src/growth/models/growth/hgp_model.py` | Reference HGP — mimic its API; the new HGPHetero will extend it. |
| `src/growth/stages/stage1_volumetric/trajectory_loader.py` | The H5 → `PatientTrajectory` loader. Extend, do not rewrite. |
| `src/growth/stages/stage1_volumetric/gompertz.py` | Gompertz mean function — reuse for HGP_Gompertz_Hetero. |
| `experiments/stage1_volumetric/run_stage1.py` | Existing orchestrator — read it carefully and clone its structure for the new orchestrator. Do not modify it (keep as the homoscedastic baseline). |
| `experiments/utils/experiment_output.py` | `save_stage_results`, `save_experiment_metadata` — reuse verbatim. |

**H5 schema you will read from** (from `merge_predictions.py` and
`convert_mengrowth_to_h5.py`):

```text
MenGrowth.h5 (v2.0)
├── attrs: {n_scans, n_patients, version="2.0", domain="MenGrowth",
│           dataset_type="longitudinal", uncertainty_rank=32,
│           ensemble_source="r32_M20_s42"}
├── images           [N_scans, 4, 192, 192, 192] float32      ← unused here
├── segs             [N_scans, 1, 192, 192, 192] int8         ← unused here
├── scan_ids         [N_scans] str
├── patient_ids      [N_scans] str
├── timepoint_idx    [N_scans] int32                          ← ordinal time
├── semantic/
│   ├── volume       [N_scans, 4]  float32  log1p volumes
│   ├── location     [N_scans, 3]  float32
│   └── shape        [N_scans, 3]  float32  [sphericity, enh_ratio, infiltr.]
├── longitudinal/
│   ├── patient_offsets  [N_patients+1] int32  CSR offsets
│   └── patient_list     [N_patients] str
├── splits/{lora_train, lora_val, test}    ← unused here
├── metadata/
│   ├── grade        [N_scans] int8                ← currently -1
│   ├── age          [N_scans] float32             ← mostly NaN
│   ├── sex          [N_scans] str                 ← mostly "unknown"
│   └── study_date   [N_scans] str    YYYY-MM-DD   ← present where available
└── uncertainty/                                     ← THE NEW INPUT FOR THIS WORK
    ├── attrs: {n_members=20, rank=32, seed=42, source_csv=...}
    ├── vol_mean              [N_scans] float32   ensemble mean V_ET (mm³)
    ├── vol_std               [N_scans] float32   ensemble std V_ET (mm³)
    ├── logvol_mean           [N_scans] float32   ⟨log(V_ET+1)⟩  ← USE THIS
    ├── logvol_std            [N_scans] float32   sqrt of var ←  AND THIS
    ├── vol_median            [N_scans] float32   robust mean (V)
    ├── vol_mad               [N_scans] float32   robust scale (V)
    ├── logvol_median         [N_scans] float32   robust mean (log V)
    ├── logvol_mad            [N_scans] float32   robust scale (log V), raw MAD
    ├── logvol_mad_scaled     [N_scans] float32   1.4826 × MAD   ← ROBUST σ
    ├── vol_ensemble          [N_scans] float32   V from majority-voted seg
    ├── logvol_ensemble       [N_scans] float32   log(V_ensemble+1)
    ├── mean_entropy          [N_scans] float32   voxel-wise H̄
    ├── mean_mi               [N_scans] float32   voxel-wise I(y; θ)
    ├── mean_var              [N_scans] float32   voxel-wise variance
    ├── men_mean_entropy      [N_scans] float32   masked to MEN class
    ├── men_mean_mi           [N_scans] float32
    ├── men_boundary_entropy  [N_scans] float32   boundary voxels only
    ├── men_boundary_mi       [N_scans] float32
    └── per_member_volumes    [N_scans, 20] float32   raw vol per member
```

**Key choice — which `(ŷ, σ̂²)` pair do we use?**

| Choice | `ŷ_ij`            | `σ̂²_v,ij`                | Rationale |
|--------|-------------------|---------------------------|-----------|
| A (default) | `logvol_mean`    | `logvol_std**2`           | Gaussian MLE on log-scale; matches the LLN result we already have for the ensemble (advisor input). |
| B (robust)  | `logvol_median`  | `logvol_mad_scaled**2`    | Median/MAD; robust to ensemble outliers. |
| C (mask-of-means) | `logvol_ensemble` | `logvol_std**2`        | Use the volume of the majority-voted segmentation as the point estimate; keep the ensemble dispersion as the variance. Justified when the union/majority mask is the clinical handle. |

Default to **A**. Make B and C selectable from the YAML config under
`uncertainty.estimator`.

**Reduce-and-replace memo for the agent.** The H5 already contains everything
you need to convert into `PatientTrajectory`. Do **not** recompute volumes from
the segmentation maps and do **not** re-run the ensemble. The
`semantic/volume[:, 3]` column is `log(V_ET + 1)` from the *majority-voted*
segmentation; for the heteroscedastic pipeline, replace it with
`uncertainty/{logvol_mean, logvol_std}`.

---

## 2. Theoretical foundation — uncertainty propagation in LME

> Read this section in full before writing `lme_hetero.py`. The model is **not**
> equivalent to a weighted MixedLM, and it is **not** equivalent to inflating the
> residual variance to a constant CV. The mechanism below is the formulation in
> Carroll, Ruppert, Stefanski & Crainiceanu (2006), §9.2; Pinheiro & Bates
> (2000), §5.2; and Bates, Mächler, Bolker & Walker (2014), §1.1. References at
> the end of the section.

### 2.1 Model

Let patient $i$ have $n_i$ observations $(t_{ij}, y_{ij}, \sigma^2_{v,ij})$ where
$y_{ij} = \log(V_{ij}+1)$ is the noisy log-volume and $\sigma^2_{v,ij}$ is the
**known** segmentation-derived variance of $y_{ij}$. The data-generating
process is

$$
\boxed{
y_{ij} \;=\; \underbrace{(\beta_0 + u_{0i}) + (\beta_1 + u_{1i})\,t_{ij}}_{\text{latent log-volume } f(t_{ij};\,\boldsymbol\beta,\mathbf u_i)}
\;+\; \underbrace{\eta_{ij}}_{\text{biological residual}}
\;+\; \underbrace{\xi_{ij}}_{\text{measurement noise}}
}
$$

with

$$
\mathbf u_i = (u_{0i},u_{1i})^\top \sim \mathcal N(\mathbf 0, \boldsymbol\Omega),
\quad
\boldsymbol\Omega = \begin{pmatrix}\tau_0^2 & \rho\tau_0\tau_1 \\ \rho\tau_0\tau_1 & \tau_1^2\end{pmatrix},
$$

$$
\eta_{ij} \sim \mathcal N(0,\sigma_n^2)\quad \text{i.i.d.},\qquad
\xi_{ij} \sim \mathcal N(0,\sigma_{v,ij}^2)\quad \text{independent and \emph{known}}.
$$

The two error terms are independent of each other and of the random effects.

### 2.2 Marginal covariance per patient

Stack $\mathbf y_i \in \mathbb R^{n_i}$, $X_i \in \mathbb R^{n_i\times 2}$ with
columns $\mathbf 1, \mathbf t_i$, and $Z_i = X_i$ (random-intercept-and-slope).
Define

$$
R_i \;\equiv\; \sigma_n^2 \, I_{n_i} \;+\; \mathrm{diag}(\sigma^2_{v,i,1},\,\ldots,\,\sigma^2_{v,i,n_i}),
\qquad
V_i \;\equiv\; Z_i\,\boldsymbol\Omega\,Z_i^\top + R_i.
$$

Then $\mathbf y_i \mid \boldsymbol\beta,\boldsymbol\theta \sim \mathcal N(X_i\boldsymbol\beta,\,V_i)$
with $\boldsymbol\theta = (\sigma_n^2,\,\tau_0^2,\,\tau_1^2,\,\rho)$.

### 2.3 REML criterion

Profile out the fixed effects via GLS:

$$
\hat{\boldsymbol\beta}(\boldsymbol\theta) \;=\; \Bigl(\sum_i X_i^\top V_i^{-1} X_i\Bigr)^{-1}\sum_i X_i^\top V_i^{-1}\mathbf y_i.
$$

The restricted log-likelihood is

$$
\ell_{\text{REML}}(\boldsymbol\theta)
\;=\;-\tfrac12\sum_i\!\Bigl[\,\log\lvert V_i\rvert + (\mathbf y_i - X_i\hat{\boldsymbol\beta})^\top V_i^{-1}(\mathbf y_i - X_i\hat{\boldsymbol\beta})\Bigr]\;-\;\tfrac12\log\!\Bigl|\sum_i X_i^\top V_i^{-1}X_i\Bigr|.
$$

This is differentiable in $\boldsymbol\theta$. Optimise with `scipy.optimize.minimize`
(L-BFGS-B). Use the parameterisation
$\boldsymbol\theta = (\log\sigma_n^2,\,\log\tau_0^2,\,\log\tau_1^2,\,\mathrm{atanh}(\rho))$
to enforce positivity and $|\rho|<1$ without bound constraints. Pass analytical
gradients only if straightforward; finite differences are acceptable given
$|\boldsymbol\theta|=4$.

> **Numerical detail.** Each $V_i$ is at most $6\times 6$ in our cohort
> (max 6 timepoints per patient). Cholesky-based solves are essentially free.
> Total cost per likelihood evaluation: $\mathcal O(N\,\bar n^3) = \mathcal O(33\cdot 30) \approx 1\,000$ flops.

### 2.4 BLUP — the propagation mechanism

Conditional on $\boldsymbol\theta,\hat{\boldsymbol\beta}$, the empirical Bayes
estimator of the patient-specific random effects is

$$
\boxed{\;
\hat{\mathbf u}_i \;=\; \boldsymbol\Omega\, Z_i^\top\, V_i^{-1}\,(\mathbf y_i - X_i\hat{\boldsymbol\beta})
\;}
$$

with posterior covariance
$\widehat{\mathrm{Var}}(\mathbf u_i\mid \mathbf y_i) = \boldsymbol\Omega - \boldsymbol\Omega Z_i^\top V_i^{-1} Z_i \boldsymbol\Omega$.

**Why this is propagation.** The matrix $V_i^{-1}$ multiplies the residuals. As
$\sigma^2_{v,ij}\to\infty$ the $j$-th row/column of $V_i^{-1}$ collapses, so
that observation contributes nothing to $\hat{\mathbf u}_i$. As
$\sigma^2_{v,ij}\to 0$ the contribution is maximal. This is the per-scan
Bayesian shrinkage that the homoscedastic LME cannot represent.

### 2.5 Predictive distribution at a new time $t^\star$

Two regimes — both must be supported:

**(a) Latent-trajectory prediction** (what is the *true* log-volume?):

$$
\hat y^\star_{i,\text{latent}} = (\hat\beta_0+\hat u_{0i}) + (\hat\beta_1+\hat u_{1i}) t^\star,
\quad
\mathrm{Var}^\star_{\text{latent}} = \mathbf z^{\star\top}\, \widehat{\mathrm{Var}}(\mathbf u_i)\,\mathbf z^\star + \sigma_n^2,
$$

with $\mathbf z^\star = (1,t^\star)^\top$.

**(b) Observable prediction** (what will the segmentation ensemble report?):

$$
\mathrm{Var}^\star_{\text{obs}} = \mathrm{Var}^\star_{\text{latent}} + \sigma^{2\star}_v,
$$

where $\sigma^{2\star}_v$ is either (i) the held-out scan's measured
`logvol_std**2` if available (**post-hoc calibration**), or (ii) the predicted
$\sigma^2_v$ from morphology if a meta-model is later added (out of scope for
this spec).

For LOPO calibration, the held-out scan's `logvol_std**2` is known and **must**
be added (regime b). For decision-making (e.g. exceedance probability), use
regime a.

### 2.6 References

- Laird, N. M. & Ware, J. H. (1982). *Random-effects models for longitudinal data*. **Biometrics**, 38, 963–974.
- Robinson, G. K. (1991). *That BLUP is a good thing: the estimation of random effects*. **Statistical Science**, 6, 15–32.
- Pinheiro, J. C. & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*, Ch. 5 — `varFunc`, heteroscedastic residuals.
- Carroll, R. J., Ruppert, D., Stefanski, L. A. & Crainiceanu, C. M. (2006). *Measurement Error in Nonlinear Models: A Modern Perspective* (2nd ed.), Ch. 9 — longitudinal and mixed-effects measurement-error models.
- Bates, D., Mächler, M., Bolker, B. & Walker, S. (2014). *Fitting Linear Mixed-Effects Models Using lme4*. arXiv:1406.5823. The conditional formulation $\mathcal Y\mid\mathcal B \sim \mathcal N(X\beta + Zb, \sigma^2 W^{-1})$ with known weights $W$ is the lme4 mechanism that we generalise here to additive (rather than multiplicative) measurement-error.
- Schielzeth, H. et al. (2020). *Robustness of linear mixed-effects models to violations of distributional assumptions*. **Methods in Ecology and Evolution**, 11, 1141–1152.
- Engelhardt, J. et al. (2023). *Evaluation of four tumour growth models to describe the natural history of meningiomas*. **eBioMedicine**, 91, 104570 — the meningioma NLME with population-level CV; see §3.4 for the per-scan generalisation we are implementing.
- Behbahani, M. et al. (2024). **Neuro-Oncology Practice**, 11(1), 14–21 — replicates Engelhardt with automatic segmentation.

---

## 3. Theoretical foundation — uncertainty propagation in GP

### 3.1 Model

A scalar GP with mean $m(t)$, covariance $k(t,t';\boldsymbol\phi)$, biological
noise $\sigma_n^2$, and **known** per-observation noise $\sigma_{v,ij}^2$:

$$
f \sim \mathcal{GP}(m,\,k),\qquad
y_{ij}\mid f \sim \mathcal N\bigl(f(t_{ij}),\;\sigma_n^2 + \sigma_{v,ij}^2\bigr).
$$

For pooled training data $\mathbf X\in\mathbb R^{N_\text{tot}\times 1}$,
$\mathbf y\in\mathbb R^{N_\text{tot}}$, define
$\boldsymbol\Sigma = \sigma_n^2 I + \mathrm{diag}(\sigma^2_{v,1},\ldots,\sigma^2_{v,N_\text{tot}})$.

### 3.2 Posterior — closed form

Standard GP posterior with augmented diagonal noise:

$$
\boxed{
\hat f(t^\star) \;=\; m(t^\star) + \mathbf k_\star^\top (K + \boldsymbol\Sigma)^{-1}(\mathbf y - \mathbf m),
}
$$

$$
\mathrm{Var}[f(t^\star)\mid\mathbf y] \;=\; k(t^\star,t^\star) - \mathbf k_\star^\top (K + \boldsymbol\Sigma)^{-1}\mathbf k_\star.
$$

Predictive variance for the *observed* y-target adds $\sigma_n^2 + \sigma^{2\star}_v$
(same regime distinction as §2.5).

### 3.3 Marginal likelihood

$$
\log p(\mathbf y\mid \mathbf X,\boldsymbol\phi,\sigma_n^2)
= -\tfrac12(\mathbf y - \mathbf m)^\top(K+\boldsymbol\Sigma)^{-1}(\mathbf y-\mathbf m)
  -\tfrac12\log|K+\boldsymbol\Sigma| - \tfrac{N_\text{tot}}{2}\log 2\pi.
$$

Optimise $\boldsymbol\phi$ (kernel hyperparameters: signal variance $\sigma_f^2$,
lengthscale $\ell$) and $\sigma_n^2$ by ML-II; the per-observation noise
$\sigma^2_{v,ij}$ is **fixed**.

### 3.4 Conditioning per held-out patient

Identical to the existing `ScalarGP.predict`: condition the fitted GP on the
held-out patient's observations using a fresh `GPRegression`/sklearn instance
with the patient's $\sigma^2_v$ values on the diagonal and the same kernel
hyperparameters. This is the per-patient personalised posterior.

### 3.5 Implementation choice

Two practical realisations exist, both correct:

| Library | Class | Per-obs noise | Biological noise |
|---|---|---|---|
| `sklearn.gaussian_process.GaussianProcessRegressor` | `alpha=array` | array on diagonal | additional `WhiteKernel` term |
| `GPy.models.GPHeteroscedasticRegression` | `het_Gauss.variance.fix(σ²_v)` | fixed array | extra `Bias`/`White` term in kernel; or absorb into `het_Gauss.variance` |

> **Decision.** Use **sklearn** for `ScalarGPHetero` and `HGPHetero`. Reasons:
> (1) the `alpha=array` interface is unambiguous and well-documented
> ([scikit-learn API reference](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html));
> (2) it composes cleanly with `WhiteKernel` for the biological noise;
> (3) it avoids GPy's idiosyncratic `Y_metadata` API for the heteroscedastic
> likelihood, which is error-prone when conditioning per-patient.
>
> Keep GPy for the homoscedastic baselines as-is. The two implementations will
> produce numerically identical posteriors when the kernels and noise terms
> match (sanity-checked in tests; see §10).

### 3.6 References

- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, MIT Press, §2.2 (closed-form posterior with general noise covariance).
- Goldberg, P. W., Williams, C. K. I. & Bishop, C. M. (1998). *Regression with input-dependent noise: A Gaussian process treatment*. **NIPS** 10, 493–499.
- Kersting, K., Plagemann, C., Pfaff, P. & Burgard, W. (2007). *Most likely heteroscedastic Gaussian process regression*. **ICML**.
- Schulam, P. & Saria, S. (2015). *A framework for individualizing predictions of disease trajectories by exploiting multi-resolution structure*. **NeurIPS**.

---

## 4. Theoretical foundation — propagation in HGP

The Hierarchical GP (`HGPHetero`) decomposes a trajectory as

$$
y_i(t) = m(t) + g_i(t) + \epsilon_i(t),
$$

with $m$ either linear (LME-derived) or Gompertz (curve_fit-derived). The
heteroscedastic version treats the per-scan residual variance the same as in
§3.1: $\epsilon_i(t)\sim\mathcal N(0,\sigma_n^2 + \sigma_{v,i}^2(t))$. The
mean function $m$ is fitted on the **same** per-observation precision-weighted
data as the new heteroscedastic LME (so the population mean is itself
uncertainty-aware), and the residual GP $g_i$ uses the heteroscedastic noise
on residuals. The implementation reuses `LMEHeteroGrowthModel` or a
weighted `curve_fit` (with `sigma=σ_v`) for the Gompertz case.

---

## 5. Time variable — `--real-time` flag

The H5 has `metadata/study_date` (YYYY-MM-DD). When present, compute

$$
\Delta_{ij} \;=\; \mathrm{date}(t_{ij}) - \mathrm{date}(t_{i,1})\quad\text{[days]},
$$

per patient (baseline = first scan). When absent for any patient in the cohort,
**do not silently fall back**; instead, log a warning at INFO level listing the
patients with missing dates and either (a) skip those patients or (b) use the
ordinal time for them and the real time for the rest, configurable via the
YAML key `time.missing_date_strategy ∈ {"skip","mixed","fail"}`. Default:
`"mixed"`.

Existing `trajectory_loader.py` already supports `time_variable="days_from_baseline"`
but expects a pre-computed `time_delta_days` dataset. Extend it to derive
deltas from `metadata/study_date` when `time_delta_days` is absent. Concretely:

```python
def _compute_deltas_from_dates(
    study_dates: np.ndarray,         # [N_scans] str (YYYY-MM-DD or "")
    patient_offsets: np.ndarray,     # [N_patients+1]
) -> tuple[np.ndarray, list[bool]]:
    """Return (delta_days [N_scans] float64, has_dates [N_patients] bool)."""
```

Patients with any missing/malformed date in their trajectory get
`has_dates[i] = False` and their entries in the returned array are NaN. The
caller (the loader) decides what to do.

The CLI of every new orchestrator must accept `--real-time` (no value), which
translates to `time_variable: days_from_baseline` in the runtime config (CLI
overrides YAML).

---

## 6. Module-by-module specification

### 6.1 Extend `PatientTrajectory`

**File:** `src/growth/shared/growth_models.py`

Add an optional field:

```python
@dataclass
class PatientTrajectory:
    patient_id: str
    times: np.ndarray
    observations: np.ndarray
    covariates: dict[str, float] | None = None
    observation_variance: np.ndarray | None = None  # [n_i] or [n_i, D]; log-scale
```

**Backwards compatibility.** All existing models (`LMEGrowthModel`, `ScalarGP`,
`HierarchicalGPModel`) ignore this field; the heteroscedastic variants require
it. If a heteroscedastic model receives a trajectory with
`observation_variance is None`, raise `ValueError("LMEHetero requires observation_variance to be set")`.

`__post_init__` checks: shape consistency
(`observation_variance.shape == observations.shape[:1]` for scalar D=1), all
values finite and non-negative, and warn if any value is exactly zero
(suggests a degenerate ensemble and would singularise $V_i$).

**Acceptance test:** existing `test_growth_models.py` still passes; a new test
in `test_uq_propagation.py` constructs a trajectory with `observation_variance`
and round-trips it through `save_trajectories` / `load_trajectories`.

### 6.2 Extend the trajectory loader

**File:** `src/growth/stages/stage1_volumetric/trajectory_loader.py`

Add a new function (do not break the existing one):

```python
def load_uncertainty_trajectories_from_h5(
    h5_path: str | Path,
    *,
    time_variable: str = "ordinal",            # "ordinal" | "days_from_baseline"
    estimator: str = "mean_std",               # "mean_std" | "median_mad" | "mask_mean"
    exclude_patients: list[str] | None = None,
    min_timepoints: int = 2,
    covariate_features: list[str] | None = None,
    semantic_covariates: list[str] | None = None,
    skip_all_zero_volume: bool = True,
    missing_date_strategy: str = "mixed",      # "skip" | "mixed" | "fail"
    floor_variance: float = 1e-6,              # avoid singular V_i
) -> list[PatientTrajectory]:
    """Load trajectories with per-observation log-volume variance.

    Reads `uncertainty/logvol_*` and (optionally) `metadata/study_date`,
    returns trajectories with `observation_variance` populated.
    """
```

Behaviour:

1. Open H5 read-only. Validate `f.attrs["version"] == "2.0"` and
   `"uncertainty" in f` (raise informative error if not).
2. Pull `(ŷ, σ̂²_v)` per the `estimator` argument:
   - `"mean_std"`     → `logvol_mean`, `logvol_std**2`
   - `"median_mad"`   → `logvol_median`, `logvol_mad_scaled**2`
   - `"mask_mean"`    → `logvol_ensemble`, `logvol_std**2`
3. Apply `np.maximum(σ̂²_v, floor_variance)` to prevent singular $V_i$.
4. Compute time deltas if `time_variable=="days_from_baseline"`:
    a. Try `time_delta_days` dataset first.
    b. Else try `metadata/study_date` and compute deltas.
    c. Apply `missing_date_strategy`.
5. Build trajectories using the existing CSR offsets + sort logic. Reuse
   `_build_covariates` and `_build_semantic_covariates` private helpers.
6. Re-export from `src/growth/stages/stage1_volumetric/__init__.py`.

**Acceptance criteria:**

- New unit test loads a real H5 (use the small `--max-patients` artefact,
  see §10) and asserts that the returned list has `observation_variance`
  populated, all values are finite, non-negative, and that
  `len(traj.observation_variance) == traj.n_timepoints`.
- A test with a synthetic H5 that has zero `logvol_std` for one scan asserts
  that the floor is applied.
- A test with `time_variable="days_from_baseline"` and a synthetic
  `metadata/study_date` asserts the deltas are correct (e.g. 2024-01-01 →
  2024-04-01 = 91 days).

### 6.3 `LMEHeteroGrowthModel`

**File:** `src/growth/models/growth/lme_hetero.py`
**Reference homoscedastic counterpart:** `lme_model.py`

```python
class LMEHeteroGrowthModel(GrowthModel):
    """LME with per-observation known measurement-error variance.

    Custom REML implementation. The residual covariance is

        R_i = σ_n² · I_{n_i} + diag(σ²_v,i,1, ..., σ²_v,i,n_i)

    where σ²_n is fitted and σ²_v are read from
    PatientTrajectory.observation_variance. Hyperparameters
    (σ²_n, τ²_0, τ²_1, ρ) are optimised by L-BFGS-B over a log/atanh
    parameterisation; fixed effects are profiled out by GLS.

    Args:
        method: "reml" (default) or "ml".
        n_restarts: Number of L-BFGS-B restarts from random initialisations.
        max_iter: Maximum iterations per restart.
        seed: Random seed.
        use_covariates: Whether to extend the fixed-effect design with covariates.
        covariate_names: Ordered names of covariates.
        missing_strategy: How to handle missing covariates ("skip", "impute_mean", "drop_patient").
        floor_variance: Minimum allowed σ²_v entry (passed through).
    """

    def fit(self, patients): ...
    def predict(self, patient, t_pred, n_condition=None): ...
    def name(self): ...
```

**Implementation notes (do these things, in this order)**:

1. Validate that every patient has `observation_variance` set with correct shape;
   error informatively otherwise.
2. Build the global design from each patient's $X_i = (\mathbf 1, \mathbf t_i)$
   stacked. Covariates extend $X_i$ but **not** $Z_i$ (random effects stay on
   intercept + slope only — the cohort is too small for more random effects).
3. Define `_neg_reml(theta_unconstrained)`:
    a. Unpack to $(\sigma_n^2, \tau_0^2, \tau_1^2, \rho)$ via $\exp$ / $\tanh$.
    b. Build $\boldsymbol\Omega$ from $\tau_0^2,\tau_1^2,\rho$.
    c. For each patient $i$: build $V_i$, compute $L_i = \mathrm{chol}(V_i)$,
       compute $\log|V_i| = 2\sum\log\mathrm{diag}(L_i)$ and the GLS sufficient
       statistics $X_i^\top V_i^{-1} X_i$ and $X_i^\top V_i^{-1} \mathbf y_i$.
    d. Sum across patients, solve for $\hat{\boldsymbol\beta}$, evaluate $\ell_{\text{REML}}$.
    e. Return $-\ell_{\text{REML}}$ (and the fixed-effect estimate as a
       side-effect for use after optimisation).
4. Run `scipy.optimize.minimize` with `method="L-BFGS-B"`, `n_restarts=5`
   different random initialisations
   $\sigma_n^2 \sim \mathrm{LogU}(10^{-3},10^0)$,
   $\tau_0^2,\tau_1^2 \sim \mathrm{LogU}(10^{-2},10^1)$,
   $\rho \sim \mathrm U(-0.5,0.5)$. Pick the run with the highest
   log-likelihood.
5. Return a `FitResult` with the optimised hyperparameters in
   `hyperparameters` (in the natural, not log, parameterisation), the
   condition number of the largest $V_i$, and the converged flag.

**Prediction (`.predict`)**:

Given a held-out patient with observations $\mathbf y_i$ at times $\mathbf t_i$
(use the first `n_condition` if given), measurement variances $\boldsymbol\sigma^2_v$,
and prediction times $\mathbf t^\star$:

1. Build $X_i, Z_i, V_i$ using the optimised $(\sigma_n^2,\boldsymbol\Omega)$.
2. Compute $\hat{\mathbf u}_i$ and its posterior covariance per §2.4.
3. For each $t^\star$:
    a. $\mathbf z^\star = (1, t^\star)^\top$.
    b. Latent mean: $\hat\beta_0 + \hat u_{0i} + (\hat\beta_1 + \hat u_{1i}) t^\star$.
    c. Latent variance: $\mathbf z^{\star\top} \widehat{\mathrm{Var}}(\mathbf u_i) \mathbf z^\star + \sigma_n^2$.
    d. **For LOPO calibration** (the default), add the held-out scan's
       known $\sigma^{2\star}_v$ to the variance.
4. 95% interval: $\hat y \pm 1.96\sqrt{\mathrm{Var}}$.

`PredictionResult.variance` should be the full observable variance (regime b).
Add a `latent_variance` field via a separate API or kwarg if a downstream
consumer needs regime a; for now, expose **both** as an additional dict in
`PredictionResult.metadata` (extend `PredictionResult` with an optional
`metadata: dict | None = None` field if you don't already have one).

**Edge cases:**

- $n_i < 2$: should be filtered out at trajectory load time, but defensively
  raise `ValueError("LMEHetero requires n_i >= 2")`.
- Fall back to a random-intercept-only model (set $\tau_1^2 = \rho = 0$) if
  the random-slope optimisation fails, mirroring the homoscedastic LME's
  fallback. Log a warning at WARNING level.

**Acceptance criteria** (these are tests the agent must add):

- **Recovers homoscedastic LME when σ²_v ≡ 0.** With `observation_variance`
  set to a small constant for every observation (effectively zero), the
  `LMEHetero` fit should be numerically close (within 1% RMSE on log-volume
  predictions and within 5% on each variance component) to
  `LMEGrowthModel`.
- **Down-weights noisy scans.** On a 5-patient synthetic dataset (linear
  trajectories with σ_n² known and one patient given a single observation
  with σ²_v = 100×σ_n²), the BLUP for that patient must be closer to the
  population mean than the BLUP from the homoscedastic LME. Check
  $|\hat u_{1i,\text{hetero}}| < |\hat u_{1i,\text{homo}}|$.
- **Closed-form sanity check.** For a single patient with two observations and
  a degenerate prior $\boldsymbol\Omega \to 0$, the predictive mean must equal
  the precision-weighted average of the two observations. This is a
  one-equation closed-form result; assert numerical equality to $10^{-6}$.

### 6.4 `ScalarGPHetero`

**File:** `src/growth/models/growth/scalar_gp_hetero.py`
**Reference homoscedastic counterpart:** `scalar_gp.py`

API mirrors `ScalarGP` but the constructor adds:

- `floor_variance: float = 1e-6` — applied to incoming `observation_variance`.
- `noise_var_bounds: tuple[float, float] = (1e-6, 5.0)` — bounds for
  $\sigma_n^2$ (biological noise term).

Internals:

- Use `sklearn.gaussian_process.GaussianProcessRegressor` with
  `alpha=σ²_v_pooled` (an array of length $N_\text{tot}$) and kernel
  `ConstantKernel(σ²_f) * Matern(length_scale=ℓ, nu=2.5) + WhiteKernel(σ²_n)`.
  The `WhiteKernel` provides the *additional* biological noise on top of the
  per-obs `alpha`; this is the cleanest way to fit $\sigma_n^2$ jointly with
  $\sigma^2_f, \ell$.
- Mean function: replicate the existing `_residualize` covariate handling and
  the linear/constant/zero mean-function options. For `mean_function="linear"`,
  fit $a, b$ jointly with the kernel by demeaning $\mathbf y$ with a small
  prior fixed-effect regression (use the existing OLS residualisation pattern
  from `scalar_gp.py`).
- Per-patient conditioning: build a fresh sklearn GPR with the same kernel
  and `alpha=σ²_v_patient`, fit on the patient's data with frozen
  hyperparameters (set `optimizer=None` and pass the fitted theta), then
  predict with `return_std=True`. To freeze hyperparameters, pass
  `kernel.set_params(...)` with `kernel.theta` from the trained model and
  set the bounds to fixed.
  - Alternative simpler path: implement the conditioning manually with the
    closed-form posterior in §3.2. This is **the recommended path** because
    sklearn's hyperparameter freezing API is clunky and the math is trivial:

```python
def _condition_and_predict(
    K_train: np.ndarray,         # [n_train, n_train] without noise
    Sigma: np.ndarray,           # [n_train] diagonal noise (σ_n² + σ²_v)
    y_train: np.ndarray,         # [n_train]
    K_star: np.ndarray,          # [n_pred, n_train]
    K_star_star: np.ndarray,     # [n_pred] diag(k(t*, t*))
    sigma_v_star: np.ndarray,    # [n_pred] for regime (b); zeros for (a)
    sigma_n_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, variance) for the predictive distribution."""
    L = np.linalg.cholesky(K_train + np.diag(Sigma))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mean = K_star @ alpha
    v = np.linalg.solve(L, K_star.T)
    var_latent = K_star_star - np.einsum("ij,ij->i", v.T, v.T) + sigma_n_sq
    return mean, var_latent + sigma_v_star
```

Implement the closed form (cleaner, fewer dependencies on sklearn internals).

**Acceptance criteria:**

- **Reduces to homoscedastic GP when σ²_v ≡ 0.** As for LME (§6.3), check
  numerical equivalence to `ScalarGP` within 1% RMSE on predictions and
  within 5% on hyperparameters.
- **Calibrated coverage at 95% on a Gaussian-data synthetic.** Generate 100
  synthetic patients with known σ²_v, run LOPO-CV, and check that the
  empirical coverage at 95% is in [0.92, 0.97]. (This test goes in
  `test_uq_propagation.py` and is fairly robust given the closed form.)

### 6.5 `HGPHetero`

**File:** `src/growth/models/growth/hgp_hetero.py`
**Reference:** `hgp_model.py`

The HGP fits a GP on residuals after subtracting a population mean. The
heteroscedastic variant:

1. Fits a population mean using **`LMEHeteroGrowthModel`** (or weighted
   `curve_fit` with `sigma=σ_v` for the Gompertz mean). Pass through the
   same `observation_variance` that will be used for the GP.
2. Computes residuals $r_{ij} = y_{ij} - \hat m(t_{ij}) - \hat u_{0i}^{(\text{LME})} - \hat u_{1i}^{(\text{LME})} t_{ij}$
   (or just $y_{ij} - \hat m(t_{ij})$ if you choose the empirical-Bayes-only
   strategy used in the existing HGP — match the existing convention).
3. Fits the residual GP on `(t_{ij}, r_{ij}, σ²_{v,ij})` using the same
   closed-form posterior as `ScalarGPHetero`. Reuse code.

**Note on Gompertz.** For the Gompertz mean function, replace `curve_fit` with
weighted `scipy.optimize.curve_fit(... , sigma=σ_v_per_obs, absolute_sigma=True)`.
This automatically uses the inverse-variance weights for the nonlinear
least-squares. Acceptance: a synthetic Gompertz signal with known noise
recovers parameters within 5% bias.

### 6.6 New metrics

**File:** `src/growth/shared/metrics.py`

Add the following:

```python
def compute_crps_gaussian(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> float:
    """Closed-form CRPS for Gaussian predictive distribution.

    Gneiting & Raftery (2007), JASA. With ω = (y - μ)/σ:

        CRPS = σ * [ ω(2Φ(ω) - 1) + 2φ(ω) - 1/√π ]

    Returns the mean CRPS over the batch. Lower is better.
    """

def compute_coverage_at_levels(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> dict[float, float]:
    """Empirical coverage at each nominal level (Gaussian assumption)."""

def compute_interval_score(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float,
) -> float:
    """Winkler interval score for (1-α) prediction intervals.

    IS_α = (u-l) + (2/α)·(l-y)·𝟙[y<l] + (2/α)·(y-u)·𝟙[y>u]

    Gneiting & Raftery (2007), §6.2. Lower is better.
    """
```

`compute_crps_gaussian` uses `scipy.stats.norm.cdf/pdf`. Closed-form reference:
[scoringRules vignette, CRAN](https://cran.r-project.org/web/packages/scoringRules/vignettes/crpsformulas.html).

Wire these into the `LOPOEvaluator`'s aggregate metrics by extending
`_compute_aggregate_metrics` to add `crps`, `is_95`, `is_80`, and per-level
coverage. **Do not break the existing keys** (`r2_log`, `mae_log`, etc.); add
new ones alongside.

### 6.7 New orchestrator

**File:** `experiments/stage1_volumetric/run_stage1_uq.py`
**Config:**  `experiments/stage1_volumetric/config_uq.yaml`

Mirrors `run_stage1.py` but:

1. Loads trajectories via `load_uncertainty_trajectories_from_h5` instead of
   `load_trajectories_from_h5`.
2. Builds **paired** model configs: each homoscedastic model has its
   heteroscedastic counterpart enabled by default.
3. Adds CLI flags: `--real-time` (sets `time.variable=days_from_baseline`),
   `--estimator {mean_std,median_mad,mask_mean}`, `--config <path>`.
4. After LOPO-CV, runs the variance-decomposition module on the paired model
   set: ScalarGP → ScalarGPHetero, LME → LMEHetero, HGP → HGPHetero. Use
   `growth.evaluation.variance_decomposition.VarianceDecomposition` already
   in the codebase. Add a paired permutation test for ΔR² and ΔCRPS.
5. Writes a comparison table to `output_dir/comparison_homo_vs_hetero.json`
   and a stdout summary table identical to the one already in
   `run_stage1.py`'s footer but with extra columns: `CRPS`, `Cov_50`,
   `Cov_80`, `Cov_90`, `Cov_95`, `IS_95`.

**Config skeleton** (the agent should clone `config.yaml` and add):

```yaml
experiment:
  name: stage1_volumetric_uq
  seed: 42

paths:
  mengrowth_h5: /media/.../MenGrowth.h5
  output_dir: experiments/stage1_volumetric/results_uq

time:
  variable: ordinal                # overridden by --real-time
  missing_date_strategy: mixed

uncertainty:
  enabled: true
  estimator: mean_std              # mean_std | median_mad | mask_mean
  floor_variance: 1.0e-6

volume:
  transform: log1p                 # already applied in H5

patients:
  exclude: [MenGrowth-0028]
  min_timepoints: 2
  skip_all_zero_volume: true

covariates:
  enabled: false
  features: [age, sex]
  missing_strategy: skip

gp:
  kernel: matern52
  mean_function: linear
  n_restarts: 5
  max_iter: 1000
  lengthscale_bounds: [0.1, 50.0]
  signal_var_bounds: [0.001, 10.0]
  noise_var_bounds: [1.0e-6, 5.0]

lme:
  method: reml
  n_restarts: 5
  max_iter: 1000

models:
  scalar_gp: true
  scalar_gp_hetero: true
  lme: true
  lme_hetero: true
  hgp: true
  hgp_hetero: true
  hgp_gompertz: true
  hgp_gompertz_hetero: true

bootstrap:
  enabled: true
  n_samples: 2000
  confidence_level: 0.95
  seed: 42

variance_decomposition:
  enabled: true
  pairs:
    - [ScalarGP, ScalarGPHetero]
    - [LME, LMEHetero]
    - [HGP, HGPHetero]
    - [HGP_Gompertz, HGP_Gompertz_Hetero]
  n_permutations: 10000
```

CLI:

```bash
~/.conda/envs/growth/bin/python -m experiments.stage1_volumetric.run_stage1_uq \
  --config experiments/stage1_volumetric/config_uq.yaml \
  --real-time \
  --estimator mean_std
```

---

## 7. Refactor of `experiments/stage1_volumetric/`

**Do not touch** `run_stage1.py` and `config.yaml` (they remain the
homoscedastic baseline). All new files live alongside them.

Final tree under `experiments/stage1_volumetric/`:

```text
experiments/stage1_volumetric/
├── config.yaml                 ← unchanged (homoscedastic baseline)
├── config_uq.yaml              ← NEW
├── run_stage1.py               ← unchanged
├── run_stage1_uq.py            ← NEW
├── run_baseline.py             ← unchanged
├── regenerate_figures.py       ← unchanged for now
├── segment.py                  ← unchanged
└── README_UQ.md                ← NEW: short README on what this is, how to run it
```

`README_UQ.md` should contain:

- One-paragraph description of the heteroscedastic pipeline.
- The math summary in three lines (link to this spec for the rest).
- Two example commands: ordinal-time and real-time runs.
- A note that the homoscedastic baseline is in `run_stage1.py`.

---

## 8. Comparison & evaluation protocol

The new orchestrator must produce a **paired comparison** that defends the
methodological contribution of the thesis. Specifically:

### 8.1 Per-model metrics (bootstrap 95% CI on each)

| Metric | Where | Why |
|---|---|---|
| `R²_log` | existing `compute_r2` | Standard fit metric |
| `MAE_log`, `RMSE_log` | existing | Magnitude error |
| `Cov_{50,80,90,95}` | new `compute_coverage_at_levels` | Calibration at 4 levels |
| `IS_95`, `IS_80` | new `compute_interval_score` | Sharpness-aware coverage (Gneiting & Raftery 2007) |
| `CRPS` | new `compute_crps_gaussian` | Strictly proper scoring rule |
| `mean_ci_width_log` | existing | Sharpness alone |

### 8.2 Paired comparisons (homoscedastic vs heteroscedastic)

For each pair $(M_\text{homo}, M_\text{hetero})$ run on identical LOPO folds:

- $\Delta R^2 = R^2(\text{hetero}) - R^2(\text{homo})$ with bootstrap CI and
  paired permutation test on per-patient errors (use existing
  `paired_permutation_test`).
- $\Delta\text{CRPS}$ same protocol.
- $\Delta\text{Cov}_{95}$ same protocol — **the cleanest test of whether the
  propagation buys us calibration that homoscedastic cannot achieve**.
- Bland–Altman plot of per-patient predicted means (tight agreement is
  expected; differences will be subtle).

### 8.3 Reporting table (markdown, written to `comparison_homo_vs_hetero.md`)

```text
| Pair                   | ΔR²   | ΔR² 95% CI       | p-perm | ΔCRPS  | ΔCov_95 |
|------------------------|-------|------------------|--------|--------|---------|
| ScalarGP → Hetero      |  ...  |  [...,  ...]    | ...    |  ...   |  ...    |
| LME → LMEHetero        |  ...  |  [...,  ...]    | ...    |  ...   |  ...    |
| HGP → HGPHetero        |  ...  |  [...,  ...]    | ...    |  ...   |  ...    |
| HGP_Gomp → Hetero      |  ...  |  [...,  ...]    | ...    |  ...   |  ...    |
```

### 8.4 What "success" looks like

The thesis claim is **calibration**, not point accuracy. Expect:

- $\Delta R^2 \in [-0.05, +0.05]$ and not significant (the propagation is
  not designed to improve point predictions; segmentation noise contributes
  ~15% CV per Engelhardt 2023, which is small relative to between-patient
  growth variability).
- $\Delta\text{Cov}_{95} > 0$ and significant (the heteroscedastic intervals
  should be **wider for noisy scans and tighter for clean scans**, yielding
  better coverage).
- $\Delta\text{CRPS} < 0$ (lower is better) and significant.

If $\Delta\text{Cov}_{95}$ is not positive and significant, **report it as a
negative result** — this is the honest scientific outcome and is itself
publishable. Do not tune the model to manufacture a positive result.

---

## 9. CLAUDE.md updates

After completing the implementation, update the project's `CLAUDE.md` Resource
Hub table at §5 to add:

- Row in the experiments table:
  `Volumetric UQ propagation | experiments/stage1_volumetric/run_stage1_uq.py | 1 | NEW`
- Row in the code table for the three new heteroscedastic model files.
- A bullet under "Key Statistical Constraints" noting that the heteroscedastic
  models add 4 hyperparameters (LMEHetero) or 3 (ScalarGPHetero / HGPHetero)
  beyond their homoscedastic counterparts, so the parameter budget is:
  - LMEHetero: 6 → 6 (the additional $\sigma_n^2$ replaces the original
    homoscedastic residual variance — no net change).
  - ScalarGPHetero / HGPHetero: 5 → 5 same logic.
  - Same parameter budget; only the **likelihood** changes.

---

## 10. Tests (must pass before merge)

Add the following test files. All under
`tests/growth/`. Use `pytest` markers `phase1` and (where applicable) `slow`.

### 10.1 `test_lme_hetero.py` (~12 tests)

| Test name | What it asserts |
|---|---|
| `test_constructor_validates_args` | Invalid `method`, `n_restarts<1` raise. |
| `test_fit_requires_observation_variance` | Trajectories without it raise. |
| `test_fit_recovers_homoscedastic_when_var_zero` | Synthetic data with σ²_v=ε → matches `LMEGrowthModel` within tolerance. |
| `test_fit_downweights_noisy_scan` | High-σ²_v scan → BLUP closer to population mean. |
| `test_fit_random_intercept_fallback` | When random-slope optimisation fails, falls back. |
| `test_predict_shape` | Returns correct array shapes. |
| `test_predict_at_training_time_recovers_blup` | Predictive mean at observed times equals $X\hat\beta + Z\hat u$. |
| `test_predict_extrapolation_grows_variance` | At $t^\star \to\infty$, latent variance → ∞. |
| `test_calibration_synthetic` | On 50 synthetic patients with known σ²_v, Cov_95 ∈ [0.90, 1.00]. |
| `test_n_condition_subset` | `n_condition=1` uses only first observation. |
| `test_serialization` | `name()` returns a stable string; `FitResult` round-trips through `to_dict`. |
| `test_two_obs_closed_form_recovery` | One-patient, two-obs case matches the closed-form precision-weighted average. |

### 10.2 `test_scalar_gp_hetero.py` (~10 tests)

Analogous to the LME tests, plus:

| Test name | What it asserts |
|---|---|
| `test_matches_sklearn_when_homoscedastic_kernel` | With constant σ²_v, our closed-form posterior matches a sklearn GPR with the same kernel. |
| `test_extrapolation_reverts_to_mean_function` | Far from data, mean → linear-mean-function value. |

### 10.3 `test_uq_propagation.py` (~6 tests)

End-to-end tests:

| Test name | What it asserts |
|---|---|
| `test_h5_loader_with_uncertainty` | Loader returns trajectories with `observation_variance` populated. |
| `test_loader_floor_variance_applied` | A scan with logvol_std=0 in the H5 gets floored. |
| `test_real_time_flag_propagates` | CLI `--real-time` produces day-based times. |
| `test_run_stage1_uq_smoke` | Smoke test on a tiny synthetic H5: orchestrator runs end-to-end without errors. |
| `test_paired_permutation_works` | Variance-decomposition pairing produces sensible numbers (no NaN). |
| `test_crps_gaussian_closed_form` | Numerical CRPS via `scipy.integrate.quad` matches our closed form to 1e-6. |

### 10.4 `test_metrics_crps.py` (~4 tests)

| Test name | What it asserts |
|---|---|
| `test_crps_gaussian_at_mean_zero_var_zero` | CRPS(N(y,0), y) = 0. |
| `test_crps_reduces_to_mae_at_var_zero` | With σ→0, CRPS → |y - μ|. |
| `test_coverage_at_known_levels` | Synthetic Gaussian residuals → coverage ≈ nominal. |
| `test_interval_score_decomposes` | $\text{IS}_\alpha$ recovers width when y is inside, plus penalties. |

---

## 11. Style & engineering rules (apply to all new code)

These restate the project's existing standards (see `CLAUDE.md` §7); flagged
here because they're easy to drift from when implementing math-heavy modules.

- **Type hints** on every public signature, including return types.
- **Google-style docstrings** with `Args:`, `Returns:`, `Raises:` sections.
- **No magic numbers**. All hyperparameters from YAML via `omegaconf` /
  the `cfg` dict.
- **Atomic functions**: keep `_neg_reml`, `_build_V_i`, `_predict_one` as
  separate private helpers under 40 lines each.
- **Custom exceptions** when validation fails. Inherit from a new
  `growth.exceptions.UncertaintyPropagationError` (create this file under
  `src/growth/exceptions.py` if it does not exist).
- **Structured logging**: use `logger = logging.getLogger(__name__)`. Log
  hyperparameters at INFO at fit time; condition numbers at DEBUG; warnings
  at WARNING for degenerate ensembles.
- **Shape assertions** at function boundaries (`assert V_i.shape == (n_i, n_i)`).
- **Deterministic seeds**: every `np.random.RandomState` instance must be
  created from `seed=cfg.experiment.seed`.
- **Module organisation**: each new model file imports only from
  `growth.shared` and `numpy/scipy/sklearn`; do **not** import from
  `growth.models.growth.lme_model` cross-importing the homoscedastic
  internals. Reusable math (e.g. covariance-building) goes into a new
  `src/growth/models/growth/_covariance_utils.py` shared helper module.
- **OOP, dataclasses, submodule organisation, explicit memory management**
  per project convention.

---

## 12. Workflow for the agent (step-by-step)

Tackle in this order, committing after each step. Each step ends with a
green test run (`pytest -m "phase1 or evaluation" -x`).

1. **Read the canonical files in §1.** Do not start coding yet.
2. **Extend `PatientTrajectory`** (§6.1). Run all existing tests; nothing
   should break.
3. **Add CRPS / coverage / interval-score metrics** (§6.6). Add their unit
   tests (§10.4). All green.
4. **Implement `_covariance_utils.py`** with the patient-level $V_i$ builder,
   Cholesky-based log-determinant, and GLS solve. Unit-test these tiny
   helpers in isolation.
5. **Implement `LMEHeteroGrowthModel`** (§6.3). Add its tests (§10.1). All
   green. **Do not move on until the homoscedastic-equivalence test passes.**
6. **Implement `ScalarGPHetero`** (§6.4). Add its tests (§10.2). All green.
   Cross-check sklearn equivalence.
7. **Implement `HGPHetero`** (§6.5). Reuse step 5 and 6 internals. Add tests.
   All green.
8. **Extend `trajectory_loader.py`** (§6.2). Add the `--real-time` plumbing
   and `metadata/study_date` derivation. Add tests (§10.3, first three).
9. **Write `run_stage1_uq.py`** and `config_uq.yaml` (§6.7). Add the
   end-to-end smoke test (§10.3, last three). All green.
10. **Run a real LOPO-CV** with the small dataset
    (`--max-patients 5` workflow). Inspect outputs:
    - `comparison_homo_vs_hetero.md` exists and has the four pairs.
    - Each model's `lopo_results.json` has the new metrics.
    - Bootstrap CIs are well-formed (lower < upper).
11. **Run on the full cohort** (the 33-patient real H5). Save results to
    `experiments/stage1_volumetric/results_uq/`. Save a
    `comparison_homo_vs_hetero.json` for downstream figure generation.
12. **Update `CLAUDE.md`** (§9).
13. **Commit** with message:
    `feat(stage1): heteroscedastic LME/GP/HGP — uncertainty propagation from LoRA-Ensemble segmentation`.

---

## 13. Acceptance criteria (must all hold for merge)

- [ ] All new files exist at the paths listed in §6 and §7.
- [ ] All new tests in §10 pass under
  `~/.conda/envs/growth/bin/python -m pytest tests/growth/test_lme_hetero.py tests/growth/test_scalar_gp_hetero.py tests/growth/test_uq_propagation.py tests/growth/test_metrics_crps.py -v`.
- [ ] All pre-existing tests continue to pass under
  `~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short`.
- [ ] `run_stage1.py` (homoscedastic baseline) produces identical numerical
  output before and after this change (regression check).
- [ ] `run_stage1_uq.py` runs end-to-end on the real H5 with the default
  config and writes:
    - `lopo_results.json` per model (8 models if all enabled),
    - `bootstrap_cis.json` per model,
    - `comparison_homo_vs_hetero.json` and `.md`,
    - `experiment_metadata.json` with config snapshot, git hash, package
      versions.
- [ ] `--real-time` produces a different `time` array than the default and
  is logged at INFO.
- [ ] `--estimator median_mad` produces a different `observation_variance`
  array than `mean_std` (verified by smoke check).
- [ ] **Honest reporting**: the `comparison_homo_vs_hetero.md` reports the
  observed sign and significance of $\Delta R^2$, $\Delta\text{Cov}_{95}$,
  $\Delta\text{CRPS}$ even if the heteroscedastic model is **worse** than the
  homoscedastic on some metric. Do not select-and-report.

---

## 14. Open issues / blind spots flagged for Mario, not the agent

> The agent should not act on these but should leave a note in
> `experiments/stage1_volumetric/results_uq/OPEN_ISSUES.md` with a 1–2 line
> summary of each, for the next iteration.

1. **Floor variance choice.** $\sigma^2_{v,ij} \geq 10^{-6}$ on log-scale ≈
   0.1% relative volume uncertainty. This is essentially zero for our
   ensemble (typical std ≈ 0.001–0.04 from the LoRA results, so the floor
   matters only for the very-clean scans). Sensitivity analysis at
   $10^{-3}, 10^{-4}, 10^{-5}, 10^{-6}$ is a v1.1 ablation, not v1.0.
2. **Random-slope identifiability with $n_i = 2$.** When most patients have
   exactly two observations (57.6% of the cohort per
   `tb_08_gp_lme.tex` §1.1), the random slope is identified only through
   between-patient variability. With heteroscedasticity adding 0
   parameters and changing only the likelihood, identifiability does not
   degrade — but check the `cov_re` condition number across folds and warn
   if any fold has $\kappa(\boldsymbol\Omega) > 10^4$.
3. **Volume vs ET vs WT.** The H5 stores `semantic/volume[:, 3]` (ET) and the
   `uncertainty/logvol_*` fields are also ET (per `merge_predictions.py`
   referencing the BraTS-MEN ET label). Verify this in the loader by reading
   the source CSV name from `uncertainty/.attrs["source_csv"]` and
   asserting it matches the expected column name.
4. **Per-scan vs per-patient noise.** A latent factor that we are not
   modelling: scanner identity. The Andalusian cohort spans multiple
   scanners. ComBat-style harmonisation could be added as a covariate, but
   that's separate work.
5. **Observable vs latent regime in the calibration metric.** §2.5 splits
   the predictive variance into latent and observable parts. The default in
   §6.3 is the observable regime (adds the held-out σ²_v at test time). For
   the *clinical exceedance probability* the latent regime is what should
   be reported; expose it via `PredictionResult.metadata["latent_variance"]`
   so a downstream plotting script can pick either.

---

## 15. Minimal sanity-check commands the agent should run before declaring done

```bash
# Activate env
ENV=~/.conda/envs/growth/bin/python

# Run all relevant tests
$ENV -m pytest tests/growth/ -m "not slow and not real_data" -v --tb=short

# Smoke-run new orchestrator on a small slice of the H5
$ENV -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml

# Same with real time
$ENV -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml \
    --real-time

# Same with robust estimator
$ENV -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml \
    --estimator median_mad

# Verify regression on homoscedastic baseline (output should be byte-identical
# modulo timestamps in the JSON)
$ENV -m experiments.stage1_volumetric.run_stage1 \
    --config experiments/stage1_volumetric/config.yaml
```

All four commands must exit 0. The first runs the test suite; the next three
exercise the three CLI paths specified by the user requirements.

---

## End of spec

If anything in this document conflicts with what you find in the code, the
**code takes precedence** — flag the conflict in `OPEN_ISSUES.md` and proceed
with the codebase's convention. If the code is silent on a question, this spec
is authoritative.
