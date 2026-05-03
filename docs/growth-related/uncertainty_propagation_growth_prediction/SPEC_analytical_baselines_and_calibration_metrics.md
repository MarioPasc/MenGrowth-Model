# Stage 1 — Analytical NLME Baselines & Calibration-Aware Metrics

**Spec version:** v1.0 (2026-05-03)
**Author:** Mario Pascual González (in collaboration with Claude)
**Target:** local Claude Code agent operating on
`/home/mpascual/research/code/MenGrowth-Model/`
**Conda env:** `~/.conda/envs/growth/bin/python`
**Test command:** `~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short`
**Predecessor spec:** `SPEC_uncertainty_propagated_volume_prediction.md` (heteroscedastic LME/GP/HGP — already implemented and passing).

---

## 0. TL;DR for the agent

Add **three nonlinear-mixed-effects (NLME) analytical baselines** — Exponential,
Logistic, and Gompertz — fitted with a **population-level residual variance**
following the Engelhardt et al. (2023) convention. Add the
**Dawid–Sebastiani Score (DSS)** and the **logarithmic score (NLPD)** to the
metric suite, plus a **PIT-histogram** diagnostic plot. Wire all three new
models and three new metrics into the existing UQ orchestrator
(`run_stage1_uq.py`) so the comparison happens on identical LOPO-CV folds.

The thesis question this spec exists to answer:

> *"Do heteroscedastic LME/GP/HGP models with per-scan ensemble-derived
> uncertainty produce more reliable predictive intervals than classical
> analytical NLME models that assume a fixed population-level residual?"*

The answer is testable on a single number: $\Delta\text{DSS} =
\text{DSS}_{\text{hetero}} - \text{DSS}_{\text{NLME}}$. If $\Delta\text{DSS}<0$
and significant under a paired permutation test, the propagation buys
calibration. If not, we report it as a negative result.

You **must** read this spec in full before writing code. The math in §2 is the
formulation Engelhardt 2023 used (re-derived here for clarity); the math in §3
is the canonical DSS plus a contained derivation of why DSS specifically
isolates the per-scan-vs-population-σ question.

---

## 1. Project context

### 1.1 What already exists (do not duplicate)

| Path | Status |
|------|--------|
| `src/growth/shared/growth_models.py` — `GrowthModel` ABC, `PatientTrajectory`, `FitResult`, `PredictionResult` | Done. `PatientTrajectory.observation_variance` is in place from the prior spec. |
| `src/growth/shared/lopo.py` — `LOPOEvaluator` | Done. Reuse without modification. |
| `src/growth/shared/metrics.py` — `compute_r2`, `compute_mae`, `compute_rmse`, `compute_calibration`, `compute_crps_gaussian`, `compute_coverage_at_levels`, `compute_interval_score` | Done from previous spec. **Extend** by adding DSS + NLPD + PIT helpers. |
| `src/growth/shared/bootstrap.py` — `bootstrap_metric`, `paired_permutation_test` | Done. |
| `src/growth/models/growth/lme_model.py`, `scalar_gp.py`, `hgp_model.py` | Homoscedastic baselines. Do not modify. |
| `src/growth/models/growth/lme_hetero.py`, `scalar_gp_hetero.py`, `hgp_hetero.py` | Heteroscedastic propagation models from prior spec. Do not modify. |
| `src/growth/stages/stage1_volumetric/gompertz.py` — `GompertzMeanFunction`, `fit_gompertz` | Already used by `HierarchicalGPModel(mean_function="gompertz")`. **Reuse the curve definition**, do not re-derive the parameterisation. |
| `experiments/stage1_volumetric/run_stage1_uq.py` — orchestrator | Done. **Extend** to register the new analytical models and surface the new metrics. |
| `experiments/stage1_volumetric/config_uq.yaml` | Done. **Extend** with an `analytical:` section. |

### 1.2 What this spec adds

| Path | Action |
|------|--------|
| `src/growth/models/growth/nlme_analytical.py` | NEW. Contains `AnalyticalNLMEModel` ABC + three concrete classes. |
| `src/growth/models/growth/_nlme_internals.py` | NEW. Private module — Laplace approximation, joint negative log-likelihood, gradient/Hessian helpers, parameter packing/unpacking. |
| `src/growth/shared/metrics.py` | EDIT. Add `compute_dawid_sebastiani`, `compute_log_score`, `compute_pit`, `compute_pit_histogram`. |
| `src/growth/shared/calibration_plots.py` | NEW. Plotting helpers for PIT histogram and reliability diagram (used by the orchestrator). |
| `experiments/stage1_volumetric/run_stage1_uq.py` | EDIT. Register the three NLME models, surface DSS / NLPD in the comparison table, generate the PIT-histogram figure. |
| `experiments/stage1_volumetric/config_uq.yaml` | EDIT. New `analytical:` block. |
| `tests/growth/test_nlme_analytical.py` | NEW. Unit tests per model. |
| `tests/growth/test_metrics_dss.py` | NEW. Unit tests for DSS, NLPD, PIT. |
| `tests/growth/test_uq_propagation.py` | EDIT. Add a paired-comparison smoke test across hetero-vs-NLME. |
| `CLAUDE.md` | EDIT. Resource hub rows + a new line under "Key Statistical Constraints". |

**Files you must not touch in this iteration:**
`run_stage1.py`, `config.yaml` (homoscedastic baseline), all `*_hetero.py`
files, the LOPO evaluator, the bootstrap module, the homoscedastic LME/GP/HGP
files. They are upstream dependencies for this work; modifying them risks
breaking the heteroscedastic results we already validated.

---

## 2. Theoretical foundation — analytical NLME baselines

### 2.1 Three growth ODEs and their closed-form solutions

Following Benzekry et al. (2014), Engelhardt et al. (2023), and the thesis
Related Work §[growth-models]:

| Model | ODE | Closed-form $V(t)$ |
|---|---|---|
| Exponential | $dV/dt = a\,V$ | $V(t) = V_0\, e^{a t}$ |
| Logistic | $dV/dt = a\,V(1 - V/K)$ | $V(t) = K \big/ \bigl(1 + (K/V_0 - 1) e^{-a t}\bigr)$ |
| Gompertz | $dV/dt = a\,V \ln(K/V)$ | $V(t) = V_0 \exp\!\bigl(\tfrac{a}{b}(1 - e^{-b\,t})\bigr)$ where $b = a/\ln(K/V_0)$ |

The Gompertz form used here matches `src/growth/stages/stage1_volumetric/gompertz.py`
in the existing codebase. **Re-use that parameterisation**; do not introduce a
second one. The relevant parameters per model are:

- Exponential: $\boldsymbol\theta = (\log V_0,\,a)$. Two scalars.
- Logistic: $\boldsymbol\theta = (\log V_0,\,a,\,\log K)$. Three scalars.
- Gompertz: $\boldsymbol\theta = (\log V_0,\,a,\,b)$. Three scalars.

Working in $\log V_0$ and $\log K$ enforces positivity. We model on the
**log-volume scale**, so the prediction equation is:

$$
y_{ij} \;=\; \log V(t_{ij};\,\boldsymbol\theta_i) + \varepsilon_{ij},
\qquad
\varepsilon_{ij} \sim \mathcal N(0,\,\sigma_{\text{pop}}^2).
$$

The constant $\sigma_{\text{pop}}^2$ is the *population-level* residual
variance — this is the **definition of "classical / analytical NLME"** in
this spec, and it's what we are testing against.

### 2.2 Equivalence with Engelhardt's "constant CV" convention

Engelhardt et al. (2023, eBioMedicine 94, 104697) and Behbahani et al. (2024,
Neurooncol Practice 11(1), 14–21) report a population-level residual measured
as a coefficient of variation $\text{CV}\approx 0.15$ on the *volume* scale:

$$
V_{ij} = V(t_{ij};\boldsymbol\theta_i)\,(1 + \delta_{ij}),
\qquad \delta_{ij}\sim\mathcal N(0,\text{CV}^2).
$$

For small CV, taking the log-transform gives

$$
y_{ij} = \log V_{ij}
= \log V(t_{ij};\boldsymbol\theta_i) + \log(1 + \delta_{ij})
\;\approx\;
\log V(t_{ij};\boldsymbol\theta_i) + \delta_{ij},
$$

so $\sigma_{\text{pop}} \approx \text{CV}$. **Working on log-scale with a
constant Gaussian residual is mathematically equivalent (to first order in
CV) to working on volume-scale with a constant CV.** This justifies the
spec's choice and aligns it with the established meningioma-growth literature.

### 2.3 NLME — random-effects parameterisation

Each patient has their own $\boldsymbol\theta_i$. Write
$\boldsymbol\theta_i = \boldsymbol\beta + \mathbf u_i$ with population
fixed-effects $\boldsymbol\beta$ and random effects
$\mathbf u_i \sim \mathcal N(\mathbf 0, \boldsymbol\Omega)$.

**Critical small-cohort decision.** Our cohort is N≈33 patients with median 3
observations per patient. Identifiability collapses if every parameter has a
random effect. Default configuration (override-able in YAML):

| Model | Fixed $\boldsymbol\beta$ | Random $\mathbf u_i$ | Free RE count |
|---|---|---|---|
| Exponential | $(\log V_0, a)$ | $(\log V_0, a)$ | 2 |
| Logistic | $(\log V_0, a, \log K)$ | $(\log V_0, a)$ — $K$ shared | 2 |
| Gompertz | $(\log V_0, a, b)$ | $(\log V_0, a)$ — $b$ shared | 2 |

This matches Vaghi et al. (2020, *PLOS Comp. Biol.*) "reduced Gompertz"
philosophy — random effect on the dominant growth parameter only. With 2
random effects $\boldsymbol\Omega$ has 3 free entries (two diagonal + one
off-diagonal); together with 1 residual variance and 2–3 fixed effects, the
parameter budget is 6–7 — comparable to the heteroscedastic LME (6) and
within the §1 statistical-adequacy bound (≤ 5–8 at N=33–58).

**Lower-budget fallback** (configurable via YAML `analytical.random_effects:`
key): random effect on $a$ only (1 random effect → 1 RE-variance + 2 FE + 1
residual = 4 parameters total). Use this if the default fails to converge
on a given fold.

### 2.4 Marginal likelihood — Laplace approximation

The patient-level marginal log-likelihood, integrating out random effects:

$$
\log p(\mathbf y_i \mid \boldsymbol\beta, \boldsymbol\Omega, \sigma^2_{\text{pop}}) \;=\;
\log \int \exp\!\Bigl[\,
\log p(\mathbf y_i \mid \boldsymbol\beta, \mathbf u_i, \sigma^2_{\text{pop}})
+ \log p(\mathbf u_i \mid \boldsymbol\Omega)
\,\Bigr]\, d\mathbf u_i.
$$

The integrand is non-Gaussian (the model is nonlinear in $\mathbf u_i$).
Laplace's method gives:

$$
\boxed{\;
\log p(\mathbf y_i) \;\approx\;
\underbrace{\log p(\mathbf y_i \mid \hat{\mathbf u}_i)}_{\text{conditional fit}}
\;+\;\underbrace{\log p(\hat{\mathbf u}_i)}_{\text{prior at mode}}
\;-\;\tfrac12 \log\det\!\Bigl(\,\tfrac{1}{2\pi}\,\mathbf H_i\,\Bigr)
\;}
$$

where $\hat{\mathbf u}_i$ is the conditional mode (i.e. the
penalised-NLS solution of the joint negative log) and
$\mathbf H_i = -\nabla^2_{\mathbf u} [\,\log p(\mathbf y_i \mid \mathbf u) + \log p(\mathbf u)\,]\big|_{\mathbf u = \hat{\mathbf u}_i}$
is the Hessian of the negative joint log-density at the mode.

**Why Laplace and not just two-stage NLS?** Two-stage approximates each
patient's parameters independently and then estimates $\sigma_{\text{pop}}$
from residuals — it ignores the random-effects prior during individual
fitting and biases $\sigma_{\text{pop}}$ downward (Davidian & Giltinan 1995,
*Nonlinear Models for Repeated Measurement Data*, §5.3). For honest
comparison with our heteroscedastic models, we need a real NLME. Laplace is
the standard approximation (`nlme::nlme()` in R, `MixedLM` in NONMEM with
`METHOD=LAPLACE`); see Lindstrom & Bates (1990) and Pinheiro & Bates (2000),
§7.

### 2.5 Optimisation — outer/inner structure

```
outer:  L-BFGS-B over (β, vech(L_Ω), log σ²_pop)         where L_Ω = chol(Ω)
        for each evaluation:
inner:    for each patient i:
              find  û_i = argmax [ log p(y_i | u_i) + log p(u_i | Ω) ]
              via L-BFGS-B over u_i
              compute Hessian H_i numerically (4-pt finite difference)
              accumulate  log p(y_i)  via §2.4 boxed equation
          return -Σ_i log p(y_i)
```

Cholesky parameterisation of $\boldsymbol\Omega$ guarantees positive
definiteness without bound constraints. Initialise outer optimisation from
a pooled-NLS warm start (fit the population mean curve to all
training data ignoring random effects, take the residual variance as initial
$\sigma_{\text{pop}}^2$, and initialise $\mathbf L_\Omega = 0.1 \cdot I$).

**Wall-clock estimate.** N=33 patients, max 6 obs/patient, RE-dim=2.
Inner L-BFGS-B converges in ~10 iterations × 33 patients ≈ 330 inner steps
per outer evaluation. Numerical Hessian: 8 extra evaluations of the joint log
per patient = 264 extra evaluations. Outer L-BFGS-B converges in ~30
iterations × ~6 line-search steps = 180 outer evaluations. Total ≈ 60k
joint-log evaluations × O(N_total × n_params) flops ≈ low millions.
Per-fold fit: under 30 s on a single CPU. Full LOPO-CV: ~15 minutes per
model. This is acceptable.

### 2.6 Prediction for held-out patients

LOPO holds out patient $i^*$. Given the optimised population
$(\hat{\boldsymbol\beta}, \hat{\boldsymbol\Omega}, \hat\sigma^2_{\text{pop}})$
and the patient's conditioning observations $\mathbf y_{i^*}^{\text{cond}}$ at
times $\mathbf t_{i^*}^{\text{cond}}$:

1. **Find $\hat{\mathbf u}_{i^*}$** by maximising
   $\log p(\mathbf y_{i^*}^{\text{cond}} \mid \mathbf u) + \log p(\mathbf u \mid \hat{\boldsymbol\Omega})$
   over $\mathbf u$ via L-BFGS-B (same inner solver as in fitting).
2. **Compute posterior covariance** at the mode:
   $\widehat{\mathrm{Var}}(\mathbf u_{i^*} \mid \mathbf y_{i^*}^{\text{cond}}) = \mathbf H_{i^*}^{-1}$.
3. **Predictive mean** at query times $\mathbf t^\star$:
   $\hat y(t^\star) = \log V(t^\star; \hat{\boldsymbol\beta} + \hat{\mathbf u}_{i^*})$.
4. **Predictive variance** via the delta method:
   $$
   \mathrm{Var}\bigl[\hat y(t^\star)\bigr] \;\approx\;
   \mathbf g^\top \widehat{\mathrm{Var}}(\mathbf u_{i^*}) \mathbf g
   \;+\; \hat\sigma^2_{\text{pop}},
   $$
   where $\mathbf g = \nabla_{\mathbf u} \log V(t^\star; \hat{\boldsymbol\beta} + \mathbf u)\big|_{\hat{\mathbf u}_{i^*}}$
   has dimension equal to the random-effects dimension. Compute $\mathbf g$
   numerically (2-pt finite difference) for simplicity.

This is the standard NLME predictive distribution under the linear-Gaussian
approximation around the mode (Pinheiro & Bates 2000, §7.5).

### 2.7 Why this is a *fair* baseline

The classical NLME has access to:
- The full per-patient longitudinal sequence.
- A population-level residual $\sigma_{\text{pop}}^2$ that is fitted from data,
  not assumed.
- A flexible nonlinear mean function with appropriate random effects.

It does **not** have access to:
- The per-scan segmentation uncertainty $\sigma^2_{v,ij}$.

That is the only difference between this baseline and `LMEHetero`. The
comparison therefore isolates the contribution of per-scan uncertainty
propagation. Anything more (e.g. fitting the analytical NLME on a different
volume target, or with different time variables) breaks the comparison.

### 2.8 References

- Lindstrom, M. J. & Bates, D. M. (1990). *Nonlinear Mixed Effects Models for Repeated Measures Data*. **Biometrics**, 46(3), 673–687.
- Pinheiro, J. C. & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*, Ch. 7 — nonlinear mixed-effects.
- Davidian, M. & Giltinan, D. M. (1995). *Nonlinear Models for Repeated Measurement Data*, Chapman & Hall.
- Wolfinger, R. (1993). *Laplace's approximation for nonlinear mixed models*. **Biometrika**, 80(4), 791–795.
- Benzekry, S. et al. (2014). *Classical mathematical models for description and prediction of experimental tumor growth*. **PLOS Comp. Biol.**, 10(8), e1003800.
- Engelhardt, J. et al. (2023). *Evaluation of four tumour growth models to describe the natural history of meningiomas*. **eBioMedicine**, 94, 104697.
- Behbahani, M. et al. (2024). **Neuro-Oncology Practice**, 11(1), 14–21.
- Vaghi, C. et al. (2020). *Population modeling of tumor growth curves and the reduced Gompertz model improve prediction of the age of experimental tumors*. **PLOS Comp. Biol.**, 16(2), e1007178.

---

## 3. Theoretical foundation — calibration metrics

The previous spec already added CRPS, multi-level coverage, and Winkler
interval scores. This section adds the metric that **directly answers the
thesis question**.

### 3.1 The Dawid–Sebastiani Score (DSS)

For a predictive distribution $F$ summarised by mean $\mu_F$ and variance
$\sigma_F^2$, the DSS is

$$
\boxed{\;
S_{DS}(F, y) \;=\;
\frac{(y - \mu_F)^2}{\sigma_F^2} \;+\; \log(\sigma_F^2).
\;}
$$

Lower is better. Original definition: Dawid & Sebastiani (1999), *Coherent
dispersion criteria for optimal experimental design*, **Annals of Statistics**
27(1), 65–81. Properness review: Gneiting & Raftery (2007), JASA, §4.5.

**Properness.** $S_{DS}$ is *strictly proper* relative to the class of
Gaussian distributions and *proper* (non-strictly) relative to the wider
class of distributions characterised by their first two moments. Strict
properness ensures that a forecaster minimises expected score by reporting
the true mean and variance.

**Decomposition for Gaussian forecasts.** Substituting
$y \mid F \sim \mathcal N(\mu_F + \delta, \tau^2)$ (i.e. the truth has bias
$\delta$ relative to the forecast and variance $\tau^2$):

$$
\mathbb E[S_{DS}] \;=\; \frac{\delta^2 + \tau^2}{\sigma_F^2} + \log(\sigma_F^2).
$$

Differentiating in $\sigma_F^2$ and setting to zero:
$\sigma_F^{2,\star} = \delta^2 + \tau^2$. **The optimal predictive variance
equals the squared bias plus the true noise variance.**

**Why this directly answers the thesis question.** Consider two forecasters
on a held-out scan $j^*$ with true segmentation noise variance
$\sigma^{2,\text{true}}_{v,j^*}$ that varies scan-to-scan:

| Forecaster | $\sigma_F^2$ used | Excess penalty per scan |
|---|---|---|
| NLME (population CV) | $\bar\sigma^2_{\text{pop}}$ (constant) | $\frac{\sigma^{2,\text{true}}_{v,j^*}}{\bar\sigma^2_{\text{pop}}} + \log\bar\sigma^2_{\text{pop}}$ |
| LMEHetero (per-scan) | $\sigma^2_n + \sigma^2_{v,j^*}$ | $\frac{\sigma^{2,\text{true}}_{v,j^*}}{\sigma^2_n + \sigma^2_{v,j^*}} + \log(\sigma^2_n + \sigma^2_{v,j^*})$ |

If the ensemble σ² is informative — i.e. correlated with the true scan-level
noise — then the per-scan forecaster's denominator tracks the numerator,
keeping the quadratic term close to 1 across all scans. The population
forecaster's quadratic term oscillates between $\ll 1$ on noisy scans and
$\gg 1$ on clean scans. The mean DSS is **strictly minimised** when
$\sigma_F^2$ adapts per-scan, *provided the per-scan estimate is
calibrated*. This is exactly the test we want.

**Relationship to the logarithmic score.** For a Gaussian predictive density
$f(y) = \mathcal N(y; \mu_F, \sigma_F^2)$ the log-score is

$$
S_{\log}(F, y) = -\log f(y) = \tfrac12 \log(2\pi) + \tfrac12 \log\sigma_F^2 + \frac{(y - \mu_F)^2}{2\sigma_F^2}.
$$

So $S_{DS} = 2\,S_{\log} - \log(2\pi)$ — **DSS and log-score are equivalent
for Gaussian forecasts up to an affine transform** and produce identical
rankings on Gaussian models. We report both:

- **DSS** — the form in the literature most directly tied to "appropriateness
  of predicted dispersion"; reviewers from the meteorology / forecasting
  community recognise it.
- **NLPD = log-score** — the form that probabilistic-ML reviewers expect.

If a model in the suite is non-Gaussian (none currently are; all our
predictive posteriors are Gaussian by construction), DSS would still be
proper but no longer strictly proper, while NLPD on the actual density
would remain strictly proper. We keep both in the suite for robustness to
future model additions.

### 3.2 PIT histogram — visual calibration diagnostic

The probability integral transform (PIT) of an observation $y$ under a
continuous predictive CDF $F$ is

$$
\text{PIT}(y, F) = F(y).
$$

When $F$ is the true predictive distribution, PIT is uniformly distributed on
$[0, 1]$ (Dawid 1984, *Statistical theory: the prequential approach*,
JRSS A 147, 278–292). Plotting the histogram of PITs over the LOPO-CV
hold-outs:

- **U-shape**: forecaster is over-confident (intervals too narrow).
- **Hump-in-middle**: forecaster is under-confident (intervals too wide).
- **Flat**: well calibrated.
- **Tilt**: forecaster has bias.

Reference: Gneiting, Balabdaoui & Raftery (2007), *Probabilistic forecasts,
calibration and sharpness*, **JRSS B** 69(2), 243–268, §2.2. For Gaussian
forecasts:

$$
\text{PIT}(y; \mu, \sigma^2) = \Phi\!\left(\frac{y - \mu}{\sigma}\right).
$$

Compute via `scipy.stats.norm.cdf`. Plot a 10-bin histogram with a
horizontal line at the uniform expectation $1/10$ and binomial 95%
acceptance bands per bin.

### 3.3 Sharpness–calibration scatter plot

Per Gneiting et al. (2007 JRSS B), the principle is "maximise sharpness
subject to calibration". A useful diagnostic plot:

- x-axis: empirical 95% coverage (calibration).
- y-axis: mean 95% interval width (sharpness; smaller is sharper).

Each model is one point. The Pareto frontier defines the
calibration-vs-sharpness trade-off. Bootstrap each point to add 95% CI
ellipses (or rectangles).

### 3.4 The full metric panel for the thesis comparison

| Metric | Purpose | Already there? | Headline? |
|---|---|---|---|
| R² (log-volume) | point accuracy | yes | yes — verifies parity |
| MAE, RMSE (log-volume) | point error magnitude | yes | secondary |
| Coverage @ {50, 80, 90, 95}% | nominal calibration | yes | secondary |
| Mean CI width @ 95% | sharpness | yes | secondary |
| Interval Score @ {80, 95} | calibration + sharpness at fixed level | yes | secondary |
| **CRPS** | overall probabilistic forecast quality | yes | **headline** |
| **DSS** | per-scan-σ vs population-σ test | NEW | **headline (the thesis claim)** |
| NLPD (log-score) | redundant with DSS for Gaussian; for safety | NEW | secondary |
| PIT histogram | visual diagnostic (per model) | NEW | figure, not table |
| Sharpness–calibration scatter | model panel diagnostic | NEW | figure, not table |

### 3.5 References

- Dawid, A. P. & Sebastiani, P. (1999). *Coherent dispersion criteria for optimal experimental design*. **Annals of Statistics**, 27(1), 65–81.
- Dawid, A. P. (1984). *Statistical theory: the prequential approach*. **JRSS A**, 147, 278–292.
- Gneiting, T. & Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and estimation*. **JASA**, 102(477), 359–378.
- Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). *Probabilistic forecasts, calibration and sharpness*. **JRSS B**, 69(2), 243–268.
- Czado, C., Gneiting, T. & Held, L. (2009). *Predictive model assessment for count data*. **Biometrics**, 65, 1254–1261. (Decomposition diagnostics.)

---

## 4. Module-by-module specification

### 4.1 New metrics (`src/growth/shared/metrics.py`)

Add the following functions. Match the existing style — short Google
docstrings, type hints, no logging at INFO inside metric functions
(keep them pure).

```python
def compute_dawid_sebastiani(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> float:
    """Mean Dawid-Sebastiani score for Gaussian forecasts.

    S_DS(F, y) = (y - mu)^2 / sigma^2 + log(sigma^2)

    Lower is better. Strictly proper for Gaussian predictive distributions
    (Gneiting & Raftery 2007, JASA, §4.5; Dawid & Sebastiani 1999, Ann. Stat.).

    Args:
        y_true: True values, shape [N].
        mu: Predictive means, shape [N].
        sigma: Predictive standard deviations, shape [N]. Must be positive.

    Returns:
        Mean DSS over the batch.

    Raises:
        ValueError: If any sigma <= 0 or shapes mismatch.
    """


def compute_log_score(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> float:
    """Mean negative log predictive density for Gaussian forecasts.

    NLPD = -log p(y | mu, sigma^2)
         = 0.5 * [log(2π) + log(σ^2) + (y - μ)^2 / σ^2]
         = 0.5 * (DSS + log 2π)

    Lower is better. Strictly proper. For Gaussian forecasts this is rank-
    equivalent to DSS — kept separately for diagnostic redundancy and
    because reviewers from probabilistic-ML expect this name.

    Args, Returns, Raises: as compute_dawid_sebastiani.
    """


def compute_pit(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """Probability integral transform under a Gaussian forecast.

    PIT_i = Φ((y_i - μ_i) / σ_i).

    Under perfect calibration PIT ~ Uniform(0, 1). Reference:
    Gneiting, Balabdaoui & Raftery (2007), JRSS B, §2.2.

    Args:
        y_true: True values, shape [N].
        mu: Predictive means, shape [N].
        sigma: Predictive standard deviations, shape [N], all positive.

    Returns:
        PIT values, shape [N], in [0, 1].
    """


def compute_pit_histogram(
    pit_values: np.ndarray, n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Bin PIT values for a calibration histogram.

    Args:
        pit_values: PIT values, shape [N].
        n_bins: Number of bins (default 10).

    Returns:
        Dict with keys:
            'bin_edges'      [n_bins+1]  bin edges in [0, 1]
            'counts'         [n_bins]    count per bin
            'frequencies'    [n_bins]    counts / N (target = 1 / n_bins)
            'expected'       float       1 / n_bins (uniform reference)
            'binomial_ci_lo' [n_bins]    lower 95% binomial CI per bin
            'binomial_ci_hi' [n_bins]    upper 95% binomial CI per bin
    """
```

The 95% binomial CI per bin uses
`scipy.stats.binom.interval(0.95, n=N, p=1/n_bins)` divided by N. This is for
the visual reliability band on the histogram, not for a formal hypothesis
test (a Cramér-von Mises uniform test is overkill for our cohort size).

**Wire into LOPO**: extend `LOPOEvaluator._compute_aggregate_metrics`
(in `src/growth/shared/lopo.py`) to add per-protocol keys
`dss`, `log_score`, plus storing the raw PIT array under
`{protocol}/pit_values`. Backwards-compatibility test: existing keys
(`r2_log`, `mae_log`, `crps`, ...) are unchanged.

### 4.2 Calibration plotting helpers (`src/growth/shared/calibration_plots.py`)

NEW module. Two public functions:

```python
def plot_pit_histogram(
    pit_values: np.ndarray,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    title: str = "PIT histogram",
    show_ci: bool = True,
) -> plt.Axes:
    """Plot a single-model PIT histogram with binomial 95% CI bands."""


def plot_sharpness_calibration_scatter(
    coverage_per_model: dict[str, float],
    width_per_model: dict[str, float],
    coverage_ci_per_model: dict[str, tuple[float, float]] | None = None,
    width_ci_per_model: dict[str, tuple[float, float]] | None = None,
    nominal: float = 0.95,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of mean CI width vs empirical coverage at one nominal level.

    The Pareto frontier toward (nominal, 0) marks well-calibrated, sharp models.
    Models below nominal coverage are over-confident; models above with large
    width are under-confident.
    """
```

Use only matplotlib; no seaborn dependency. Match the line/marker style of
`experiments/uncertainty_segmentation/plotting/figures/fig_epistemic_diagnosis.py`
where reasonable so the thesis figures look consistent.

### 4.3 NLME internals (`src/growth/models/growth/_nlme_internals.py`)

NEW private module. Holds the math: parameter packing/unpacking, joint and
marginal log-densities, Hessians. Public surface kept minimal; the
`AnalyticalNLMEModel` subclasses are the only consumers.

```python
@dataclass
class NLMEFitState:
    """Container for the parameters of a fitted NLME model."""
    beta: np.ndarray             # fixed effects, shape [n_fixed]
    L_omega: np.ndarray          # Cholesky factor of Ω, shape [n_re, n_re]
    sigma_pop_sq: float          # residual variance on log-scale
    fixed_names: list[str]       # parameter names for `beta`
    re_names: list[str]          # subset of fixed_names that have RE


def pack_params(
    beta: np.ndarray, L_omega: np.ndarray, sigma_pop_sq: float,
) -> np.ndarray:
    """Pack into unconstrained 1-D vector for L-BFGS-B."""

def unpack_params(
    theta: np.ndarray, n_fixed: int, n_re: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inverse of pack_params."""

def patient_joint_neg_log(
    u: np.ndarray, y: np.ndarray, t: np.ndarray,
    beta: np.ndarray, L_omega: np.ndarray, sigma_pop_sq: float,
    growth_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    re_indices: np.ndarray,        # which entries of (beta + u_full) are RE
) -> float:
    """Negative log p(y|u) - log p(u|Ω) for one patient.

    growth_fn(t, theta_full) -> log_volume_predictions [n_obs].
    re_indices selects the RE-bearing components of theta.
    """

def find_mode_and_hessian(
    y: np.ndarray, t: np.ndarray,
    beta: np.ndarray, L_omega: np.ndarray, sigma_pop_sq: float,
    growth_fn: Callable, re_indices: np.ndarray, n_re: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Inner Laplace step. Returns (u_hat, H, converged)."""

def laplace_marginal_log(
    y: np.ndarray, t: np.ndarray,
    beta: np.ndarray, L_omega: np.ndarray, sigma_pop_sq: float,
    growth_fn: Callable, re_indices: np.ndarray, n_re: int,
) -> float:
    """One patient's Laplace-approximated log p(y_i)."""
```

**Numerical notes for the agent**:

1. Use `numpy.linalg.cholesky`, never `scipy.linalg.cholesky` (no benefit
   here, fewer dependencies).
2. Compute $\log\det \mathbf H$ via Cholesky if $\mathbf H$ is PD; if it
   isn't (Hessian indefinite at non-converged inner mode), return `+np.inf`
   for the patient's neg-log so the outer optimiser steps away.
3. Numerical Hessian: 4-point central difference with step
   $h = \epsilon^{1/4} \cdot (1 + \|\mathbf u\|)$ where
   $\epsilon = 10^{-12}$; standard rule from Press et al. (2007), *Numerical
   Recipes*, §5.7.
4. **Numerical stability**: clamp $\sigma_{\text{pop}}^2 \ge 10^{-6}$ inside
   the joint-log evaluation. Same for the Cholesky diagonal of
   $\boldsymbol\Omega$ — bound the diagonal via $\exp(d) + 10^{-6}$.
5. Vectorise across observations within a patient; do not vectorise across
   patients (the inner Laplace is per-patient).

### 4.4 `AnalyticalNLMEModel` ABC and three concrete classes

**File:** `src/growth/models/growth/nlme_analytical.py`

```python
class AnalyticalNLMEModel(GrowthModel, ABC):
    """Abstract NLME baseline with population-level residual variance.

    Subclasses provide the growth function (Exponential / Logistic / Gompertz)
    and the parameter names. The fitting machinery (Laplace marginal
    likelihood, L-BFGS-B outer, mode-finding inner) is shared.

    Args:
        random_effects: subset of fixed_param_names that get random effects.
            Default: per-class sensible default (see below).
        n_restarts: outer L-BFGS-B random-restart count.
        max_iter: outer iterations per restart.
        seed: random seed.
        residual_variance_init: initial σ²_pop. Default 0.04 ≈ (15% CV)².
        bound_log_v0, bound_a, bound_log_k, bound_b: hyperparameter
            bounds in the unconstrained-but-meaningful space.
    """

    @abstractmethod
    def growth_log_volume(self, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return log V(t; theta) for the given parameter vector."""

    @abstractmethod
    def fixed_param_names(self) -> list[str]:
        """Ordered list of fixed-effect parameter names."""

    @abstractmethod
    def default_random_effects(self) -> list[str]:
        """Subset of fixed_param_names with random effects (default config)."""

    def fit(self, patients: list[PatientTrajectory]) -> FitResult: ...
    def predict(
        self, patient: PatientTrajectory, t_pred: np.ndarray,
        n_condition: int | None = None,
    ) -> PredictionResult: ...
    def name(self) -> str: ...
```

The three subclasses:

```python
class ExponentialNLME(AnalyticalNLMEModel):
    """log V(t) = log V_0 + a * t.

    fixed_param_names: ["log_v0", "a"]
    default_random_effects: ["log_v0", "a"]
    """

class LogisticNLME(AnalyticalNLMEModel):
    """log V(t) = log K - log(1 + (K/V_0 - 1) * exp(-a t))
                = log K - log[1 + exp(-a t + log(K/V_0 - 1))]

    fixed_param_names: ["log_v0", "a", "log_k"]
    default_random_effects: ["log_v0", "a"]      # log_k shared
    """

class GompertzNLME(AnalyticalNLMEModel):
    """log V(t) = log V_0 + (a / b) * (1 - exp(-b * t)).

    fixed_param_names: ["log_v0", "a", "b"]
    default_random_effects: ["log_v0", "a"]      # b shared
    """
```

**Important**: do **not** ignore the existing `GompertzMeanFunction` /
`fit_gompertz` in `src/growth/stages/stage1_volumetric/gompertz.py`. Reuse
the same Gompertz formula. If the existing parameterisation differs from
the one above, prefer the existing one and update §2.1 of this spec via a
`SPEC_AMENDMENT.md` note in the same PR.

### 4.5 Wire-in to `run_stage1_uq.py`

Add to `_build_model_configs` (or wherever the orchestrator builds the
model registry):

```python
analytical_cfg = cfg.get("analytical", {})
if analytical_cfg.get("enabled", False):
    nlme_kwargs = {
        "n_restarts": analytical_cfg.get("n_restarts", 5),
        "max_iter":   analytical_cfg.get("max_iter", 1000),
        "seed":       cfg["experiment"]["seed"],
        "random_effects": analytical_cfg.get("random_effects"),  # None = default
        "residual_variance_init": analytical_cfg.get("residual_variance_init", 0.04),
    }
    if analytical_cfg.get("exponential", True):
        models["ExponentialNLME"] = (ExponentialNLME, nlme_kwargs)
    if analytical_cfg.get("logistic", True):
        models["LogisticNLME"] = (LogisticNLME, nlme_kwargs)
    if analytical_cfg.get("gompertz", True):
        models["GompertzNLME"] = (GompertzNLME, nlme_kwargs)
```

After LOPO-CV completes, extend the comparison block:

1. **Stdout summary table** — add columns `DSS`, `NLPD` to the existing
   summary; keep R², MAE, RMSE, Cov_95, mean_ci_width, CRPS already there.
2. **JSON output** — write `comparison_classical_vs_propagated.json`
   structured as:
   ```json
   {
     "groups": {
       "classical":  ["ExponentialNLME", "LogisticNLME", "GompertzNLME", "LME"],
       "propagated": ["LMEHetero", "ScalarGPHetero", "HGPHetero"]
     },
     "per_model_metrics": {<model>: {<metric>: {"value": ..., "ci_lo": ..., "ci_hi": ...}}},
     "paired_tests": [
       {"a": "GompertzNLME", "b": "LMEHetero",
        "delta_dss":  {"value": ..., "ci": [...,...], "p_perm": ...},
        "delta_crps": {...}, "delta_r2": {...}, "delta_cov_95": {...}}
     ]
   }
   ```
3. **Markdown summary** — write `comparison_classical_vs_propagated.md`:
   ```text
   ## Classical NLME vs Propagated-Uncertainty Models

   ### Per-model headline metrics (LOPO-CV, last_from_rest)

   | Model              | R²    | CRPS  | DSS   | Cov_95 | Width_95 |
   |--------------------|-------|-------|-------|--------|----------|
   | ExponentialNLME    |  ...  |  ...  |  ...  |  ...   |   ...    |
   | LogisticNLME       |  ...  |  ...  |  ...  |  ...   |   ...    |
   | GompertzNLME       |  ...  |  ...  |  ...  |  ...   |   ...    |
   | LME (homo)         |  ...  |  ...  |  ...  |  ...   |   ...    |
   | LMEHetero          |  ...  |  ...  |  ...  |  ...   |   ...    |
   | ScalarGPHetero     |  ...  |  ...  |  ...  |  ...   |   ...    |
   | HGPHetero          |  ...  |  ...  |  ...  |  ...   |   ...    |

   ### Headline paired comparisons (propagation – classical, lower better for DSS/CRPS)

   | Pair                              | ΔR²   | ΔCRPS | ΔDSS  | p (perm) | Decision |
   |-----------------------------------|-------|-------|-------|----------|----------|
   | LMEHetero – GompertzNLME          |  ...  |  ...  |  ...  |   ...    |   ...    |
   | ...                                                                                |
   ```
4. **Figures** — generate two PDFs in `output_dir/figures/`:
   - `pit_histogram_panel.pdf` — 2×4 grid (one per model).
   - `sharpness_calibration_scatter.pdf` — single panel, 7 points.

### 4.6 Config additions (`config_uq.yaml`)

```yaml
analytical:
  enabled: true
  exponential: true
  logistic: true
  gompertz: true
  n_restarts: 5
  max_iter: 1000
  residual_variance_init: 0.04        # ≈ (15% CV)^2 per Engelhardt 2023
  random_effects: null                # null → per-model default; or list e.g. ["a"]

reporting:
  pit:
    enabled: true
    n_bins: 10
  sharpness_calibration_scatter:
    enabled: true
    nominal_level: 0.95
  classical_vs_propagated:
    pairs:
      - [GompertzNLME,    LMEHetero]
      - [GompertzNLME,    ScalarGPHetero]
      - [GompertzNLME,    HGPHetero]
      - [LogisticNLME,    LMEHetero]
      - [ExponentialNLME, LMEHetero]
      - [LME,             LMEHetero]      # bridges classical and modern
    metrics: [r2_log, crps, dss, cov_95]
    n_permutations: 10000
```

---

## 5. Comparison protocol — what "success" looks like

The thesis claim is **calibration**, not point accuracy. The ranked
hypotheses to test, in order:

1. **Primary (DSS).** $\Delta\text{DSS} = \text{DSS}_{\text{LMEHetero}} - \text{DSS}_{\text{GompertzNLME}} < 0$ with paired permutation $p < 0.05$. **This is the headline thesis claim.**
2. **Secondary (CRPS).** $\Delta\text{CRPS} < 0$ with $p < 0.05$.
3. **Coverage parity.** Empirical 95% coverage closer to nominal than the
   classical model — measure absolute deviation $|\text{cov}_{95} - 0.95|$.
4. **No significant point-accuracy regression.**
   $\Delta R^2$ should not be significantly negative.

Numerical expectation, conditional on the LoRA-Ensemble uncertainty being
informative:

| Quantity | Expected sign | Magnitude |
|---|---|---|
| $\Delta R^2$ | ≈ 0 | $|·| < 0.05$, not significant |
| $\Delta\text{CRPS}$ | < 0 | small but significant |
| $\Delta\text{DSS}$ | < 0 | larger effect than CRPS — DSS is more sensitive to per-scan σ |
| $|Cov_{95} - 0.95|$ | smaller for hetero | hetero closer to nominal |
| Mean CI width | similar or smaller for hetero on average | not the right metric in isolation; must be paired with coverage |

**Honest reporting clause** (carried forward from the previous spec): if any
of these effects has the wrong sign or fails to reach significance, report it
as such in the thesis. The framework is designed to be falsifiable — that
is its scientific value. Do not tune until the desired outcome appears.

---

## 6. CLAUDE.md updates

Add to the Resource Hub `Code` table:

| Resource | Path |
|---|---|
| Analytical NLME baselines | `src/growth/models/growth/nlme_analytical.py` |
| NLME internals (Laplace) | `src/growth/models/growth/_nlme_internals.py` |
| Calibration plots | `src/growth/shared/calibration_plots.py` |

Add to Resource Hub `Experiments` table:
- (no new experiment, just an extension of `experiments/stage1_volumetric/run_stage1_uq.py`)

Add a new bullet under "Key Statistical Constraints":

> Classical NLME baselines (Exp / Logistic / Gompertz) carry 6–7 free
> parameters at N=33 (2–3 fixed effects + 2-RE Ω with 3 entries + 1 residual).
> Within the §1 budget. Random-effect dimension fixed at 2 by default
> (`log_v0`, `a`); reduce to 1 (`a` only) via the YAML config if any fold
> fails to converge.

---

## 7. Tests

Add the following test files. All use `pytest` markers `phase1` or
`evaluation` as appropriate. Total ~22 new tests.

### 7.1 `tests/growth/test_metrics_dss.py` (≈ 7 tests)

| Test name | Asserts |
|---|---|
| `test_dss_at_zero_residual_zero_variance` | DSS(N(y, ε), y) for tiny ε → −∞-like (specifically `log(ε²)` dominates negatively); and DSS(N(y, 1), y) = 0 + log(1) = 0. |
| `test_dss_strictly_proper_at_optimum` | $\sigma^{2,\star} = \delta^2 + \tau^2$ check (§3.1): generate 10000 samples from $\mathcal N(\mu+\delta, \tau^2)$ and confirm that varying $\sigma_F^2$ around $\sigma^{2,\star}$ produces a minimum at $\sigma^{2,\star}$ within Monte Carlo noise. |
| `test_dss_log_score_relationship` | $\text{DSS} = 2\,\text{NLPD} - \log(2\pi)$ to numerical precision. |
| `test_dss_ranks_per_scan_better_than_population` | Construct a synthetic case where true $\sigma^2_{v}$ varies 10× across observations; assert that the per-scan forecaster has lower mean DSS than a forecaster using $\bar\sigma^2$. **This is the existence proof for the thesis claim.** |
| `test_pit_uniform_under_calibration` | 5000 samples from $\mathcal N(0, 1)$ predicted with same: PIT should be approximately uniform (Kolmogorov-Smirnov test, $p > 0.05$). |
| `test_pit_under_overconfidence` | Predict with $\sigma_F = 0.5\,\sigma_{\text{true}}$: PIT should be U-shaped (more mass near 0 and 1 than uniform; check histogram bins). |
| `test_pit_histogram_returns_correct_bins` | Shape and content of the dict returned by `compute_pit_histogram`. |

### 7.2 `tests/growth/test_nlme_analytical.py` (≈ 12 tests)

Each model gets four tests; some shared.

| Test name | Asserts |
|---|---|
| `test_exponential_recovers_synthetic` | Generate 50 patients from a known exponential NLME with known $(\boldsymbol\beta^\star, \boldsymbol\Omega^\star, \sigma_{\text{pop}}^{2,\star})$; fit; recover within 10% relative error on $\boldsymbol\beta$, within 30% on $\sigma_{\text{pop}}^2$. |
| `test_logistic_recovers_synthetic` | Same for logistic. |
| `test_gompertz_recovers_synthetic` | Same for Gompertz. |
| `test_predict_at_training_time_recovers_observations` | For one patient with 4 observations on a clean Gompertz curve, fit then predict at the training times. RMSE should be below the residual std. |
| `test_predict_extrapolation_widens_variance` | At $t^\star \to t_{\text{train,last}} + 10\Delta t$, predictive variance grows (delta-method gradient ≠ 0). |
| `test_fit_random_effect_subset` | Configure `random_effects=["a"]` only; fit; assert that $\boldsymbol\Omega$ is 1×1 and the model still converges. |
| `test_warm_start_from_pooled_nls` | Mock the warm-start function; confirm initial outer iteration is nearly stationary if the synthetic data already follows the population curve. |
| `test_n_condition_subset` | `predict(..., n_condition=1)` uses only first observation in inner Laplace. |
| `test_population_residual_is_constant` | Predictive σ² should not depend on the held-out scan's identity beyond the delta-method gradient term — i.e. residual contribution is identical across test points. |
| `test_invalid_input_raises` | Empty patient list → `ValueError`. Patient with $n_i=1$ → defensively skip with logged warning. |
| `test_serialization` | `name()` returns a stable string; `FitResult.hyperparameters` includes all fitted population params. |
| `test_lopo_smoke_5_patients` | Tiny LOPO-CV (5 synthetic patients × 3 obs) completes for each of the three models within 60 s. |

### 7.3 `tests/growth/test_uq_propagation.py` — extend (≈ 3 new tests)

| Test name | Asserts |
|---|---|
| `test_classical_vs_propagated_paired_comparison_smoke` | Run a tiny end-to-end LOPO with 6 synthetic patients including both groups; assert `comparison_classical_vs_propagated.json` is well-formed (groups present, per_model_metrics non-empty, paired_tests has expected pair labels). |
| `test_pit_figure_generated` | Same setup, assert `pit_histogram_panel.pdf` exists and is non-empty. |
| `test_sharpness_calibration_scatter_generated` | Same, assert `sharpness_calibration_scatter.pdf` exists. |

---

## 8. Style & engineering rules (carry forward)

These are the same as the previous spec. Restated for emphasis on the
math-heavy modules:

- **Type hints** on every public signature.
- **Google-style docstrings** with `Args:`, `Returns:`, `Raises:`.
- **No magic numbers**. Bounds, init values, and tolerances come from the
  YAML config or class init args, not from code.
- **Atomic functions**. Every Laplace internal stays under 40 lines.
- **Custom exceptions**: extend
  `src/growth/exceptions.py` with `class NLMEConvergenceError(Exception)`.
  Raise on outer-optimisation failure or on patient-level Hessian
  indefiniteness that the agent cannot recover from. Log inner-step issues
  at WARNING; raise the custom exception only at the end of fit.
- **Structured logging**: `logger = logging.getLogger(__name__)`. INFO at
  fit-time hyperparameters; DEBUG for per-patient mode-finding diagnostics;
  WARNING for fallback-to-1-RE on convergence failure.
- **Shape assertions** at function boundaries.
- **Deterministic seeds**: every `np.random.RandomState(seed)` derives from
  `cfg.experiment.seed`. The `n_restarts` random initialisations use
  `seed + restart_idx`.
- **Memory**: do **not** store per-patient Hessians on the model object.
  Recompute when needed for prediction (cheap).
- **OOP**: the three concrete model classes share the `AnalyticalNLMEModel`
  ABC; no copy-paste of fit/predict logic.

---

## 9. Workflow for the agent (step-by-step)

Tackle in this order. Commit and run `pytest -m "phase1 or evaluation" -x`
after each step.

1. **Read** §§1–4 of this spec, the existing `lme_hetero.py` (for API
   conventions), and `experiments/stage1_volumetric/gompertz.py` (for the
   Gompertz parameterisation). Do not write code yet.

2. **Add metrics** (§4.1). Then unit-test (`test_metrics_dss.py`, §7.1).
   Verify the headline relationship $\text{DSS} = 2\,\text{NLPD} - \log 2\pi$.

3. **Add calibration plotting helpers** (§4.2). No new tests at this step
   (the plot-rendering tests come later in the smoke test).

4. **Implement `_nlme_internals.py`** (§4.3). Heavy unit tests at the
   *helper* level: pack/unpack roundtrip, joint-log gradient consistent with
   numerical derivative on a synthetic case, `find_mode_and_hessian` returns
   PD Hessian at the synthetic mode. **Do not move on until these helpers
   are correct.**

5. **Implement `AnalyticalNLMEModel` ABC and `ExponentialNLME`** (§4.4).
   Pass `test_exponential_recovers_synthetic` (§7.2). Exponential is the
   simplest because it's linear in $t$ on log-scale — if the framework
   doesn't recover it, nothing else will work.

6. **Implement `LogisticNLME` and `GompertzNLME`**. Pass corresponding
   `test_*_recovers_synthetic` tests. Gompertz is the most important — it
   is the headline analytical baseline (Engelhardt 2023).

7. **Wire into `run_stage1_uq.py`** (§4.5). Update `config_uq.yaml` (§4.6).
   Run on a 5-patient synthetic H5 (use the existing test fixture with
   `--max-patients 5`).

8. **Run on the real H5**. Inspect:
   - `comparison_classical_vs_propagated.{json,md}` exist and are
     well-formed.
   - `pit_histogram_panel.pdf` and `sharpness_calibration_scatter.pdf`
     render.
   - Each model's `lopo_results.json` has `dss`, `log_score`, and a
     `pit_values` array under `aggregate_metrics` / `predictions`.

9. **Sanity-check the headline result**. Open
   `comparison_classical_vs_propagated.md` and read the row
   `LMEHetero – GompertzNLME`. Report ΔDSS, ΔCRPS, p-value to Mario in the
   PR description, even if results are negative. **Do not retune to chase a
   positive result.**

10. **Update `CLAUDE.md`** (§6).

11. **Commit** with message:
    `feat(stage1): analytical NLME baselines + DSS/NLPD/PIT metrics for thesis comparison`.

---

## 10. Acceptance criteria (must all hold for merge)

- [ ] All new files exist at the paths in §1.2.
- [ ] All new tests in §7 pass under
  `~/.conda/envs/growth/bin/python -m pytest tests/growth/test_metrics_dss.py tests/growth/test_nlme_analytical.py tests/growth/test_uq_propagation.py -v`.
- [ ] All pre-existing tests continue to pass under
  `~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short`.
- [ ] `run_stage1.py` (homoscedastic baseline) numerical output unchanged
  before and after this change.
- [ ] `run_stage1_uq.py` (heteroscedastic models from prior spec) numerical
  output unchanged for those models — adding new models must not perturb
  existing ones (use a regression test on `lopo_results.json` for one of
  the existing models).
- [ ] `run_stage1_uq.py` with `analytical.enabled: true` writes:
  - `comparison_classical_vs_propagated.json`
  - `comparison_classical_vs_propagated.md`
  - `figures/pit_histogram_panel.pdf`
  - `figures/sharpness_calibration_scatter.pdf`
  - per-model `lopo_results.json` with `dss`, `log_score`, `pit_values`
    populated
- [ ] DSS rank-equivalence to NLPD verified at the metric level (test).
- [ ] On the real H5, all three NLME models converge on at least 90% of
  LOPO folds. If any model fails > 10% of folds, fall back to the 1-RE
  variant for that model and re-run; document in `OPEN_ISSUES.md`.
- [ ] **Honest reporting**: the markdown summary reports the observed sign
  and significance of every paired ΔDSS, ΔCRPS, ΔR², ΔCov_95 even when
  the heteroscedastic model is worse. Do not select-and-report.

---

## 11. Open issues (not for the agent — for Mario)

Leave a 1–2 line summary in
`experiments/stage1_volumetric/results_uq/OPEN_ISSUES.md`.

1. **Random-effects dimension.** Default 2 RE per model. If the cohort
   expands to N=58 (the userMemories aspirational target), revisit by
   adding RE on `b` for Gompertz and on `log_k` for Logistic, then
   re-run.
2. **Combined residual.** Vaghi et al. (2020) used a combined additive +
   proportional residual ($\sigma = \sigma_1 + \sigma_2 V$). Engelhardt
   (2023) used a CV-only residual. We use constant Gaussian on log-scale
   (= proportional on volume-scale to first order). A v1.1 ablation
   could compare the three residual specifications, but they are unlikely
   to materially change the conclusion.
3. **Bayesian variant.** A fully Bayesian NLME via PyMC (with weakly
   informative priors on $\boldsymbol\beta, \boldsymbol\Omega,
   \sigma_{\text{pop}}^2$) would give more faithful predictive
   distributions, especially in the tails. The Laplace approximation is a
   point estimate of the marginal likelihood — adequate for the comparison
   we want, but not the last word. Future work.
4. **Per-time test points and mid-trajectory extrapolation.** The current
   prediction protocol (`last_from_rest`, `all_from_first`) is the same
   for all models. If a clinical scenario calls for 6-month-ahead vs
   12-month-ahead prediction comparison, the protocol must be fixed
   per-comparison — do not let one model use a different protocol than
   another.
5. **PIT histogram interpretation at small N.** With N≈33 hold-outs and
   10 bins, expected count per bin is 3.3. Binomial 95% CIs overlap
   substantially with the uniform line — the visual is suggestive, not
   conclusive. Pair with the Cramér-von Mises test for uniformity if
   needed (`scipy.stats.cramervonmises_2samp` against a uniform sample of
   matched size). This is a v1.1 nicety, not v1.0.

---

## 12. Sanity-check commands the agent should run before declaring done

```bash
ENV=~/.conda/envs/growth/bin/python

# 1. Unit tests
$ENV -m pytest tests/growth/test_metrics_dss.py        -v --tb=short
$ENV -m pytest tests/growth/test_nlme_analytical.py    -v --tb=short
$ENV -m pytest tests/growth/test_uq_propagation.py     -v --tb=short
$ENV -m pytest tests/growth/                           -m "not slow and not real_data" -v --tb=short

# 2. Smoke run on a dev slice (analytical only)
$ENV -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml \
    --estimator mean_std

# 3. Full run including --real-time
$ENV -m experiments.stage1_volumetric.run_stage1_uq \
    --config experiments/stage1_volumetric/config_uq.yaml \
    --real-time

# 4. Regression check: hetero pipeline output unchanged from prior spec
diff prev_results/lopo_results_LMEHetero.json \
     experiments/stage1_volumetric/results_uq/LMEHetero/lopo_results.json
# should differ only in timestamps

# 5. Quick visual check
xdg-open experiments/stage1_volumetric/results_uq/figures/pit_histogram_panel.pdf
xdg-open experiments/stage1_volumetric/results_uq/figures/sharpness_calibration_scatter.pdf
```

All commands must exit 0. The fourth confirms that the previous
heteroscedastic results are not perturbed by the addition of the analytical
models — a critical regression check.

---

## End of spec

If the codebase contradicts this document, the **code wins** — flag the
contradiction in `OPEN_ISSUES.md` and follow the existing convention. If the
codebase is silent and this spec is also silent, default to the simpler /
more conservative option and document the choice in the function docstring.
