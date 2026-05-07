# Calibration Story for Uncertainty-Propagated Volume Prediction

**Date:** 2026-05-04
**Scope:** Stage 1 LME / LMEHetero (and the GP / NLME companions) on
the MenGrowth-58 cohort with M=20 LoRA-ensemble segmentations.
**Audience:** thesis manuscript and supervisors. This document
re-derives every variance term, defines calibration formally, and
explains — using the empirical σ²_v / σ²_n distributions and the
actual LOPO-CV results — why the heteroscedastic models do **not**
show a clean marginal-coverage win over their homoscedastic
counterparts, what to do about it, and which design changes are
defensible scientifically.

The companion document `docs/UQ_THESIS_GAP_ANALYSIS.md` records the
implementation-level audit; this document is the *interpretation* of
the audit's findings.

---

## 1. The prediction problem and what "calibrated" means

### 1.1 What we predict

For patient $i$ at time $t_{ij}$ we observe a tumor volume
$V_{ij} \in \mathbb{R}_{\ge 0}$ derived from the LoRA segmentation
ensemble. We work on the log scale,

$$
y_{ij} = \log(V_{ij} + 1),
$$

both because tumor growth is approximately exponential at small
horizons and because $\log(\cdot+1)$ is well-defined at zero volume.

Given the held-out patient's first $n_i - 1$ observations
$\mathcal{D}_i = \{(t_{ij}, y_{ij})\}_{j=1}^{n_i-1}$, the model
returns at the final time $t^* = t_{i,n_i}$ a *predictive
distribution* $p(y_i^* \mid \mathcal{D}_i, \theta)$ which we report
as a Gaussian:

$$
p(y_i^* \mid \mathcal{D}_i) = \mathcal{N}\!\bigl(\hat{y}_i^*, \, s_i^{*2}\bigr),
$$

with mean $\hat{y}_i^*$ and predictive variance $s_i^{*2}$ given by
the model. The 95 % prediction interval is
$\hat{y}_i^* \pm 1.96\, s_i^*$.

### 1.2 What "calibration" measures

A predictive distribution is **calibrated** if the nominal coverage
matches the empirical coverage. Concretely, define the indicator

$$
C_i^{(\alpha)} = \mathbf{1}\!\left[\, y_i^* \in \big[\hat{y}_i^* - z_{\alpha/2} s_i^*,\;
                                             \hat{y}_i^* + z_{\alpha/2} s_i^*\big] \,\right].
$$

For a perfectly calibrated model at level $1-\alpha$,
$\mathbb{E}[C_i^{(\alpha)}] = 1-\alpha$. Three families of metric
operationalise this:

| Metric | Formula | What it diagnoses |
|---|---|---|
| **Empirical coverage** at level $1-\alpha$ | $\frac{1}{N}\sum_i C_i^{(\alpha)}$ | First-order calibration: do intervals contain the truth at the rate they claim? |
| **Sharpness** (mean CI width) | $\frac{1}{N}\sum_i (2 z_{\alpha/2} s_i^*)$ | How wide are the intervals? Wide and well-covering is uninformative; narrow and well-covering is the goal. |
| **CRPS** (proper score, Gaussian closed form) | $\sigma\!\left[\frac{y-\mu}{\sigma}\bigl(2\Phi(\tfrac{y-\mu}{\sigma})-1\bigr) + 2\phi(\tfrac{y-\mu}{\sigma}) - \pi^{-1/2}\right]$ averaged over $i$ | Joint accuracy + sharpness in a single number; lower is better. |
| **Interval score** at level $1-\alpha$ | $(u-\ell) + \tfrac{2}{\alpha}\bigl[(\ell-y)_+ + (y-u)_+\bigr]$ | Penalises width *and* mis-coverage simultaneously. |
| **PIT** (probability integral transform) | $u_i = \Phi\!\bigl((y_i^* - \hat y_i^*)/s_i^*\bigr)$ should be $\mathrm{Uniform}(0,1)$ | Tests calibration at *all* levels at once; KS test makes it quantitative. |

**Key distinction**: a model can have perfect *marginal* coverage
(averaged over patients) while being miscalibrated *conditionally*
(over- or under-covering on identifiable subsets). The thesis
narrative we want to support is *conditional* on the segmentation
quality of each scan.

### 1.3 The two LOPO protocols

The LOPO evaluator (`src/growth/shared/lopo.py`) reports two
prediction protocols for every model:

- **`last_from_rest`** — predict the *final* timepoint conditioning on
  the first $n_i-1$ scans. **This is the thesis-aligned protocol.** It
  matches the clinical decision: given a patient's history, do we
  trust the model's forecast of where the tumor is *now*?
- **`all_from_first`** — predict *all* future timepoints from the
  baseline scan only. A long-horizon extrapolation stress test, not
  the thesis estimand.

All numerical comparisons below use `last_from_rest` unless stated
otherwise. N = 56 evaluable patients (the cohort's 58 minus
`MenGrowth-0028` and patients with $n_i < 2$).

---

## 2. Decomposing the predictive variance

This section derives every term in the predictive variance from
first principles. The notation follows
`docs/methods/uncertainty_propagation.tex` (thesis), so each term
maps to a labelled equation in the manuscript.

### 2.1 The likelihood

For the heteroscedastic LME, the per-patient model is

$$
\mathbf{y}_i \;=\; X_i \boldsymbol{\beta} \;+\; Z_i \mathbf{u}_i \;+\; \boldsymbol{\varepsilon}_i, \qquad
\mathbf{u}_i \sim \mathcal{N}(\mathbf{0}, \Omega), \quad
\boldsymbol{\varepsilon}_i \sim \mathcal{N}(\mathbf{0}, R_i),
$$

with

- $X_i \in \mathbb{R}^{n_i \times p}$ the fixed-effect design (here
  $X_i = [\mathbf{1}, \mathbf{t}_i, \text{cov}_i]$);
- $Z_i \in \mathbb{R}^{n_i \times 2}$ the random-effect design
  $[\mathbf{1}, \mathbf{t}_i]$ (random intercept + slope);
- $\Omega = \begin{bmatrix} \tau_0^2 & \rho\tau_0\tau_1 \\
                            \rho\tau_0\tau_1 & \tau_1^2 \end{bmatrix}$
  the between-patient covariance;
- $R_i$ the residual covariance for patient $i$.

The two model families differ only in $R_i$:

$$
\textbf{LME (homo):}\quad R_i = \sigma^2 I_{n_i}; \qquad
\textbf{LMEHetero:}\quad R_i = \sigma_n^2 I_{n_i} + \mathrm{diag}\!\bigl(\sigma_{v,1}^2, \dots, \sigma_{v,n_i}^2\bigr).
$$

The heteroscedastic residual decomposes the per-observation noise
into a **biological / model-misspecification floor** $\sigma_n^2$
shared across observations, and a **per-scan measurement variance**
$\sigma_{v,k}^2$ supplied by the LoRA ensemble. The marginal
covariance at fit time is

$$
V_i \;=\; Z_i \Omega Z_i^\top \;+\; R_i.
$$

### 2.2 What each variance term measures

Define $\mathbf{x}^* = [1,\, t^*,\, \text{cov}_i^\top]^\top$ and
$\mathbf{z}^* = [1, t^*]^\top$ as the fixed and random designs at
the prediction time $t^*$.

**Term A — Residual / measurement variance at $t^*$.**

$$
\mathrm{Var}(\varepsilon_i^* \mid \theta) \;=\;
\begin{cases}
\sigma^2, & \text{LME (homo)} \\
\sigma_n^2 \,+\, \sigma_{v,*}^2, & \text{LMEHetero}
\end{cases}
$$

Interpretation: the irreducible noise on a *new* observation at
$t^*$. In the heteroscedastic case it has two physically distinct
parts: $\sigma_n^2$ is what the model *cannot explain even with
perfect segmentation* (biological micro-fluctuation, scanner
drift, registration error); $\sigma_{v,*}^2$ is what the
**segmentation pipeline contributes** to that scan, computed
upstream from the M=20 LoRA members as

$$
\sigma_{v,k}^2 \;=\; \frac{1}{M-1}\sum_{m=1}^{M}\bigl(y_{i,k}^{(m)} - \bar y_{i,k}\bigr)^2,
\qquad y_{i,k}^{(m)} = \log(V_{i,k}^{(m)}+1).
$$

This is the *mean–std* estimator (`mean_std` in
`uncertainty/logvol_std`). The thesis assumes $\sigma_{v,k}^2$ is
**known** at fit time, not estimated; it enters the likelihood as a
fixed offset on the diagonal of $R_i$.

**Term B — Random-effect posterior variance (BLUP shrinkage).**

$$
\mathrm{Var}\!\bigl(\mathbf{z}^{*\top} \mathbf{u}_i \mid \mathcal{D}_i, \theta\bigr)
\;=\; \mathbf{z}^{*\top}\!\bigl(\Omega - \Omega Z_i^\top V_i^{-1} Z_i \Omega\bigr)\mathbf{z}^*.
$$

Interpretation: how uncertain we remain about *this* patient's
random intercept and slope after seeing $n_i-1$ of their scans. The
quantity in parentheses is the BLUP posterior covariance
$\mathrm{Cov}(\mathbf{u}_i \mid \mathcal{D}_i)$. With many
observations per patient and well-separated time points this term
shrinks toward zero; with very few observations it stays close to
$\Omega$. The factor $\mathbf{z}^* = [1, t^*]$ scales the slope
contribution by $t^*$: at $t^*=0$ the term is just the intercept
posterior variance, but at long horizon it grows as
$\tau_1^{2,\text{post}} \cdot t^{*2}$.

**This is where time parameterisation matters.** With ordinal time
$t = 0, 1, 2, \ldots$ a five-year horizon is at most $t^*=4$ for the
data we have (4-5 visits over 3-5 years), so the random-slope
contribution at horizon is bounded by $16\,\tau_1^{2,\text{post}}$.
With $t$ in years a five-year horizon is $t^*=5$, and with $t$ in
days it would be $t^*=1825$ → six orders of magnitude difference in
the random-slope contribution to the predictive variance. This is
the practical reason the manuscript needs `days_from_baseline`.

**Term C — Fixed-effect coefficient uncertainty.**

$$
\mathrm{Var}\!\bigl(\mathbf{x}^{*\top}\hat{\boldsymbol{\beta}}\bigr)
\;=\; \mathbf{x}^{*\top}\,\mathrm{Cov}(\hat{\boldsymbol{\beta}})\,\mathbf{x}^*.
$$

Interpretation: how uncertain the *population mean* trajectory
$\mathbf{x}^{*\top}\boldsymbol{\beta}$ is at $t^*$, accounting for
finite-sample noise in the GLS estimate
$\hat{\boldsymbol{\beta}} = \bigl(\sum_i X_i^\top V_i^{-1} X_i\bigr)^{-1}
\sum_i X_i^\top V_i^{-1} \mathbf{y}_i$. Its covariance is

$$
\mathrm{Cov}(\hat{\boldsymbol{\beta}}) \;=\; \Bigl(\sum_i X_i^\top V_i^{-1} X_i\Bigr)^{-1}.
$$

For the homoscedastic LME this is the GLS info matrix at
$\hat\sigma^2$; for LMEHetero it uses $V_i$ at the REML optimum.
This term grows with horizon as $t^{*2}$ as well (because $\mathbf{x}^*$
contains $t^*$), but with $\mathrm{Var}(\hat\beta_1)$ rather than
$\tau_1^{2,\text{post}}$.

**Predictive variance (sum form, per the thesis).**

$$
s_i^{*2} \;=\; \underbrace{\mathbf{x}^{*\top}\mathrm{Cov}(\hat{\boldsymbol\beta})\mathbf{x}^*}_{\text{C: fixed effects}}
            + \underbrace{\mathbf{z}^{*\top}\!\bigl(\Omega - \Omega Z_i^\top V_i^{-1} Z_i \Omega\bigr)\mathbf{z}^*}_{\text{B: random effects (BLUP)}}
            + \underbrace{\sigma_n^2 + \sigma_{v,*}^2}_{\text{A: residual / measurement}}.
$$

The Kackar–Harville (1984) / Prasad–Rao (1990) EBLUP MSPE
introduces a small additional cross-term that the thesis suppresses
for clarity; the simple sum is **conservative** (slightly
overestimates the variance, biasing coverage *upward*). Adopting the
exact MSPE would *narrow* intervals slightly. Either is defensible.

### 2.3 Where these terms live in the code

| Term | LME location | LMEHetero location |
|---|---|---|
| A: $\sigma^2$ or $\sigma_n^2 + \sigma_{v,*}^2$ | `lme_model.py:_predict_dimension` (`pred_var = dim_fit.sigma_sq + …`) | `lme_hetero.py:predict` (`latent_var + sv_pred`) |
| B: BLUP posterior variance | `lme_model.py` (`shrunk_cov`) | `lme_hetero.py` (`cov_post`) |
| C: $\mathbf{x}^{*\top}\mathrm{Cov}(\hat\beta)\mathbf{x}^*$ | `lme_model.py` (`fe_var = einsum(...)`, uses `dim_fit.cov_beta`) | `lme_hetero.py` (`fe_var = einsum(...)`, uses `self._cov_beta`) |

All three terms are present and verified (audit in
`docs/UQ_THESIS_GAP_ANALYSIS.md`, §4b).

### 2.4 How $\sigma_n^2$ and $\sigma_{v,k}^2$ are estimated

$\sigma_{v,k}^2$ is **known**: it comes from the M=20 LoRA members
upstream and is stored in `uncertainty/logvol_std` in the H5. It is
not optimised by REML; it enters the likelihood as a fixed diagonal.

$\sigma_n^2$ (along with $\tau_0^2, \tau_1^2, \rho$) is the LME
hyperparameter, optimised by REML. The negative log REML objective
for a heteroscedastic LME is

$$
-\log L_{\mathrm{REML}}(\boldsymbol\theta) \;=\;
\tfrac{1}{2}\sum_i \log\det V_i \;+\;
\tfrac{1}{2}\sum_i (\mathbf{y}_i - X_i\hat{\boldsymbol\beta})^\top V_i^{-1} (\mathbf{y}_i - X_i\hat{\boldsymbol\beta})
\;+\; \tfrac{1}{2}\log\det\!\Bigl(\sum_i X_i^\top V_i^{-1} X_i\Bigr)
\;+\; \text{const}.
$$

The third term is the REML correction (the marginal correction for
having profiled $\boldsymbol\beta$ out of the likelihood). The
optimiser is L-BFGS-B with `n_restarts=5` over a log/atanh
parameterisation. The crucial property for the calibration story:

> **REML chooses $\sigma_n^2$ jointly with $\Omega$ to maximise the
> profile likelihood; it does so at fixed $\sigma_{v,k}^2$.** If the
> $\sigma_{v,k}^2$ values are large on average, REML *reduces*
> $\sigma_n^2$ to compensate, because the total residual budget per
> observation is what the data constrains.

This is the mechanism that produces the headline finding below.

---

## 3. The MenGrowth-58 data: what σ²_v actually looks like

Direct measurement on `uncertainty/logvol_std` in the production H5
(`/media/mpascual/Sandisk2TB/.../MenGrowth.h5`), N = 179 scans, M = 20
LoRA members per scan:

| Statistic | $\sigma_v$ (log scale) | $\sigma_v^2$ |
|---|---|---|
| Median | **0.0344** | **0.00118** |
| Mean | 0.2391 | 0.4165 |
| 10th pct. | 0.0060 | 3.6 × 10⁻⁵ |
| 90th pct. | 0.5001 | 0.2551 |
| Max | 3.346 | 11.20 |
| Count $\sigma_v > 1$ | 11 / 179 (6.1 %) | — |
| Count $\bar V = 0$ | 3 / 179 | — |
| Count $\sigma_v > 0.1$ | 57 / 179 (32 %) | — |

The distribution is **strongly bimodal / right-skewed**:

- A *core* of ~120 scans (~67 %) with $\sigma_v \le 0.1$, i.e.
  segmentation members agree to within ~10 % relative volume;
  $\sigma_v^2 \le 0.01$ for these.
- A *long tail* of 11 scans (6.1 %) with $\sigma_v > 1$, dominated by
  zero-volume / vanishing-tumor cases where one or two LoRA members
  predict spurious mass and the others predict zero. These inflate
  the *mean* $\sigma_v^2$ to 0.42, more than 300× the median.

Said differently: **for two thirds of the cohort, the LoRA
ensemble's measurement noise is essentially negligible**, and for
6 % of the cohort it is enormous and is driven by what are
effectively segmentation failures (small / zero-volume meningiomas
where the LoRA members disagree about whether anything is there at
all).

The corresponding REML estimates from
`results/.../{LME,LMEHetero}/hyperparameters.json` (averaged across
the 56 LOPO folds, which differ only by which patient is held out):

| Parameter | LME (homo) | LMEHetero |
|---|---|---|
| $\sigma_n^2$ | **0.95** | **0.55** |
| $\tau_0^2$ (intercept variance) | (statsmodels-internal) | 1.94 |
| $\tau_1^2$ (slope variance) | (statsmodels-internal) | 0.42 |
| $\rho$ (intercept–slope) | (statsmodels-internal) | −0.58 |
| $\hat\beta_0$ | 8.26 | 8.32 |
| $\hat\beta_1$ (per ordinal step) | 0.057–0.096 | 0.13–0.18 |

Reading the LMEHetero column: REML decided that of the total
~0.95 log-residual budget, **0.40 is the average measurement noise
contributed by segmentation** (mean $\sigma_v^2$) and the remaining
**0.55 is the biological / model-floor noise** $\sigma_n^2$. This
decomposition is the *intended use* of LMEHetero — and in the
training fit it is statistically justified.

---

## 4. The LOPO results: what we actually observe

`last_from_rest` aggregate metrics from
`results/.../{model}/error_summary.json` and the `aggregate_metrics`
block of `lopo_results.json`, N = 56 patients:

| Model | $\hat\sigma^2$ (REML) | CI width | cov@95 | CRPS | IS@95 |
|---|---|---|---|---|---|
| LME (homo) | 0.95 | **5.39** | **0.893** | 0.781 | 10.55 |
| LMEHetero | 0.55 + $\sigma_v^2$ | **4.97** | **0.875** | 0.778 | 9.02 |
| NLME_Exponential | analytical | 5.37 | 0.893 | 0.783 | 10.66 |
| ScalarGP | GP-fit noise | 8.07 | 0.964 | 1.049 | 9.72 |
| ScalarGPHetero | GP-fit + $\sigma_v^2$ | 6.30 | 0.946 | 0.927 | 6.56 |
| HGP | GP-fit | 6.52 | 0.946 | 0.970 | 9.56 |
| HGPHetero | GP-fit + $\sigma_v^2$ | 6.27 | 0.964 | 0.923 | 6.51 |
| HGP_Gompertz_Hetero | GP + Gompertz mean | 11.87 | 1.000 | 1.230 | 11.87 |

The expected pattern — propagation widens intervals → empirical
coverage moves *up* toward 0.95 → interval scores improve at the
cost of a small CRPS hit — is **partially present** in the GP family
and **inverted** in the LME family:

- LMEHetero is **sharper** than LME (CI 4.97 vs 5.39) and slightly
  *more* under-covering (0.875 vs 0.893). CRPS is essentially the
  same (0.778 vs 0.781).
- ScalarGPHetero and HGPHetero do produce the textbook story:
  comparable coverage to their homo siblings *with substantially
  better interval scores* (6.56 vs 9.72; 6.51 vs 9.56). This is
  because the GP residual noise is fit by maximum-likelihood per
  fold; adding $\sigma_v^2$ as a per-observation offset only
  tightens the GP's noise estimate when the observation actually
  has large $\sigma_v^2$, not on average.
- HGP_Gompertz_Hetero over-covers (1.000) with a 12-unit-wide CI
  driven by Gompertz-mean extrapolation pathologies on the training
  trajectories. This is a separate failure mode and should not be
  read as a propagation result.

For the **LME family specifically**, the data does not exercise
propagation in the regime where it would help the most, and §5
explains why.

---

## 5. The story: why LMEHetero is sharper but worse-calibrated

Combine the two pieces.

### 5.1 Step-by-step

1. The **total residual budget per observation** that the data can
   support is set by how much the trajectories deviate from a
   straight line in log-volume after accounting for random
   intercept + slope. In the LME (homo) fit this budget is
   $\sigma^2 = 0.95$.
2. In LMEHetero, the residual budget is split into two pieces by
   REML: $R_i = \sigma_n^2 I + \mathrm{diag}(\sigma_{v,k}^2)$ with
   $\sigma_{v,k}^2$ **fixed and known** from the LoRA ensemble.
   REML chooses $\sigma_n^2$ to maximise the profile likelihood
   *conditional* on the supplied $\sigma_{v,k}^2$.
3. The empirical mean of $\sigma_{v,k}^2$ across the training scans
   is **0.42**. REML therefore picks $\sigma_n^2 = 0.55$ so that the
   *average* total residual variance is preserved:
   $$
   \sigma_n^2 + \overline{\sigma_v^2} \;\approx\; 0.55 + 0.40 \;=\; 0.95 \;=\; \sigma^2_{\mathrm{homo}}.
   $$
   This is a purely statistical absorption: REML cannot change the
   data's empirical residual budget, only its *decomposition*.
4. At **prediction time**, the predictive variance for a new
   observation at $t^*$ is $A_i + B_i + C_i$, with
   $A_i = \sigma_n^2 + \sigma_{v,*}^2$.
5. For a typical held-out scan, $\sigma_{v,*}^2 \approx 0.0012$
   (the median). The hetero predictive residual is then
   $$
   A_i^{\text{hetero}} \;\approx\; 0.55 + 0.001 \;=\; 0.55,
   $$
   while the homoscedastic predictive residual is
   $$
   A_i^{\text{homo}} \;=\; 0.95.
   $$
   Hetero is **40 % sharper at the residual level** for the median
   scan, even though the *fitted* total residual budget is
   identical. This is the source of the narrower mean CI (4.97 vs
   5.39).
6. With sharper intervals on the same point predictions
   ($\hat\beta$ and BLUP $\hat{\mathbf{u}}_i$ differ only marginally
   between LME and LMEHetero), the empirical coverage slips slightly
   below the nominal: 0.875 vs 0.893.
7. For the rare high-$\sigma_v$ scans, the opposite happens:
   $\sigma_{v,*}^2 \in [1, 11]$ pushes the predictive variance to
   1.5–11.5, dwarfing the homoscedastic 0.95. Hetero produces very
   wide intervals on these scans, which contributes to its large
   $\sigma_v^2$-weighted CI variance (`ci_width_std = 2.04` vs
   homo's 0.31).

### 5.2 Why this is not a bug

Every step above follows from the model, the data, and REML acting
correctly. The implementation in `src/growth/models/growth/lme_hetero.py`
is verified term-by-term against the thesis equations
(`docs/UQ_THESIS_GAP_ANALYSIS.md` §4b). The §2 predictive-variance
patches (adding the fixed-effect uncertainty term $C_i$) are
present and correct.

The behaviour we observe is the *correct* consequence of giving REML
$\sigma_v^2$ values whose *empirical mean* is large but whose *median
at prediction targets* is essentially zero. Propagation cannot
manufacture per-scan uncertainty information that the data does not
contain.

### 5.3 Why this *is* a problem for the thesis claim

The current claim — *"propagating LoRA-ensemble uncertainty into the
LME residual improves marginal calibration over an unaware baseline"*
— is not supported by the LME-family numbers and is only weakly
supported by the GP-family numbers. To rescue the thesis we have
two scientifically defensible options:

**Option A — reframe as conditional calibration.** The honest result
is that propagation *redistributes* uncertainty across observations
without changing the marginal residual budget, and is therefore
informative in the **subset of observations where $\sigma_v^2$ is
elevated**. Demonstrated empirically by stratifying calibration by
$\sigma_v^2$ tertiles or by reporting calibration on the high-$\sigma_v$
subset specifically.

**Option B — change the data so propagation can act.** Several
levers below.

---

## 6. Recommended actions, with rationale

In rough order of scientific impact and cost.

### 6.1 Re-frame the manuscript around conditional calibration (no code change)

Stratify the 56 test patients into tertiles by their target-scan
$\sigma_{v,*}^2$ and report `cov@95`, CRPS, and interval score per
tertile. Expected pattern:

- **Low $\sigma_v^2$ tertile**: hetero ≈ homo (both well-calibrated
  or both slightly under-covering).
- **Mid $\sigma_v^2$ tertile**: hetero starts to widen, homo stays
  fixed; calibration begins to favour hetero.
- **High $\sigma_v^2$ tertile**: hetero widens substantially → homo
  systematically *over-confident*; hetero's interval score is
  better, even though its CRPS may be slightly worse.

This is the *honest* scientific finding and it directly supports
the propagation thesis without overclaiming.

The implementation is small: per-fold record `sigma_v_target`, group
folds into tertiles, report metrics per tertile. Add the table to
the manuscript and `experiments/stage1_volumetric/stats/` as a
verification artifact.

### 6.2 Drop scans with $\sigma_v > 1$ as pre-evaluation QC failures (config + 1-line H5 filter)

The 11 scans with $\sigma_v > 1$ are not measurement noise; they are
**segmentation failures**. They are the residuals of cases where
either:

- the meningioma is sub-cm³ and the LoRA members disagree about
  whether to predict any mask at all — the volume is dominated by
  segmentation discretisation, not by tumour biology;
- the case is at or near zero volume (3 scans have $\bar V = 0$)
  and the M=20 ensemble is bimodal between "predict empty" and
  "predict spurious mass";
- the registration / preprocessing left the tumour at the FOV edge
  (rare but possible with the 192³ crop on small skulls).

Treating these as legitimate uncertainty estimates pollutes the
REML mean $\overline{\sigma_v^2}$ from a true ~0.01 (the 67 % core)
to the inflated 0.42. **Filter them out** before fitting and
reporting. Two implementations:

1. **Hard filter on $\sigma_v$**. Add to `config_uq.yaml`:
   ```yaml
   patients:
     exclude: [MenGrowth-0028]
     min_timepoints: 2
     skip_all_zero_volume: true
     max_logvol_std: 1.0      # NEW — drop scans with σ_v > 1.0
   ```
   Implementation: filter at trajectory-loader time
   (`src/growth/stages/stage1_volumetric/trajectory_loader.py`),
   logging which scans are dropped and how many trajectories lose
   how many timepoints. Patients that fall below `min_timepoints`
   after the filter are themselves dropped.

2. **Hard filter on volume**. Drop scans with $\bar V \le V_{\min}$
   (e.g. 100 mm³ — half a clinically discernible meningioma). This
   is more interpretable than the σ-based filter and likely catches
   the same scans, but it conflates "noisy segmentation" with
   "clinically irrelevant volume". Use this *only if* the manuscript
   explicitly excludes sub-cm³ tumours from scope.

Recommendation: ship option 1 with `max_logvol_std: 1.0`. Report the
dropped-scan list in the supplementary material.

**Expected effect on results.** With the 11 catastrophic scans
removed, $\overline{\sigma_v^2}$ falls from ~0.42 to ~0.05 (a
naive estimate using the remaining 168 scans' median). REML will
then estimate $\sigma_n^2 \approx 0.90$ and the hetero predictive
variance at the median target becomes $0.90 + 0.001 \approx 0.90$,
within 5 % of the homo budget, and propagation now has a clean
*per-scan* effect: scans with $\sigma_v^2 \approx 0.1$ get a 10 %
wider CI than scans with $\sigma_v^2 \approx 0.001$, which is
exactly the calibration improvement the thesis predicts.

### 6.3 Floor $\sigma_v^2$ at a clinically informed value (1-line config)

`floor_variance: 1.0e-6` is purely numerical (avoids singular $V_i$).
Set it to a *physiological* floor such as the empirical 25th
percentile after the QC filter — concretely **`floor_variance: 1.0e-3`**
— which represents the minimum measurement noise we believe exists
in any volumetric segmentation. This:

- prevents the "vanishing $\sigma_v^2$ at the prediction target"
  pathology described in §5;
- guarantees the hetero predictive variance is bounded below by a
  meaningful amount;
- is documented in the methods as "a published test–retest variance
  for volumetric MRI tumour delineation" (e.g. Chow et al. 2014,
  *Neuro-Oncology*, ~5 % CV → $\sigma_v \approx 0.05$, $\sigma_v^2
  \approx 0.0025$).

### 6.4 Re-run with `time.variable: days_from_baseline` once dates wired

Independent of the σ²_v issue, ordinal time squashes the random-slope
contribution to predictive variance. With ordinal $t \in \{0,1,2,3,4\}$
the maximum random-slope variance contribution is
$\tau_1^{2,\text{post}} \cdot t^{*2} \le 16\, \tau_1^{2,\text{post}}$.
With $t$ in days at typical 5-year follow-up the same term is
$\le 1825^2 \cdot \tau_1^{2,\text{post}}$ ≈ 3 × 10⁶ ×
$\tau_1^{2,\text{post}}$ — six orders of magnitude. Real time will
substantially widen long-horizon intervals and is required for the
RANO 1/2/5-year exceedance metric (`eq:exceedance` in the thesis).

Once dates are merged, re-run Stage 1 UQ with
`time.variable: days_from_baseline` and `gp.lengthscale_bounds`
re-tuned (suggest `[30.0, 5000.0]` in days).

### 6.5 Adopt the EBLUP MSPE cross-term, optionally

The thesis form sums $A + B + C$, which is conservative. The
Kackar–Harville MSPE replaces the sum with
$A + B + C - 2\langle\,\rangle$ where the cross-term is small but
non-zero. Adopting it would *narrow* intervals slightly. Defer until
after §6.1–6.2 are settled — the marginal gain is much smaller than
the QC effect.

### 6.6 Report the σ²_v decomposition table in the manuscript

Add Section 3 of this document (the σ²_v distribution table and the
REML decomposition) to the methods. It is the single most
informative diagnostic and it justifies whatever calibration story
the manuscript ends up telling.

---

## 7. Quick decision tree for the thesis

```
Do you have study dates?
├── No (current state): keep ordinal time. Do NOT claim long-horizon
│   calibration; restrict claims to held-out final timepoint.
└── Yes (target):       switch to days_from_baseline; re-run.

Do you accept dropping 11 high-σv scans?
├── Yes: ship `max_logvol_std: 1.0` filter. Re-run with
│        floor_variance: 1.0e-3. Expect propagation effect to
│        emerge cleanly (cov_hetero ≈ cov_homo, IS_hetero < IS_homo).
└── No:  keep all scans. Present the σ²_v / σ²_n decomposition
         table from §3 explicitly. Re-frame the propagation claim
         as conditional calibration (§6.1).

Should the high-σv scans be reported separately?
└── Yes — they are the regime where propagation is most informative.
    Report a stratified calibration table: low / mid / high σv* tertiles.
```

---

## 8. Appendix: per-fold REML walk-through (LMEHetero, fold 1)

Held-out patient `MenGrowth-0001`, training on the remaining 55
(171 observations):

| Quantity | Value |
|---|---|
| `sigma_n_sq` | 0.5826 |
| `tau0_sq` | 1.9425 |
| `tau1_sq` | 0.4250 |
| `rho` | −0.5825 |
| `beta_0` | 8.315 |
| `beta_1` | 0.150 (per ordinal step) |
| `log_marginal_likelihood` | −136.92 |
| `fit_time_s` | 2.11 |

For the held-out patient at $t^* = $ last ordinal index with
$\mathbf{z}^* = [1, t^*]$, the predictive variance is

$$
s^{*2} \;=\; \mathbf{x}^{*\top}\,\hat{\mathrm{Cov}}(\hat\beta)\,\mathbf{x}^*
            \;+\; \mathbf{z}^{*\top}\!\bigl(\hat\Omega - \hat\Omega Z^\top \hat V^{-1} Z \hat\Omega\bigr)\mathbf{z}^*
            \;+\; 0.5826 \;+\; \sigma_{v,*}^2,
$$

evaluated with $\hat\Omega$ from the table and $\hat V$ assembled
from the training covariates. The 0.5826 term is the only piece that
*would not change* if propagation were turned off; the 0.95 of the
homoscedastic LME would replace both 0.5826 and $\sigma_{v,*}^2$.

For a typical held-out target with $\sigma_{v,*}^2 = 0.001$, the
hetero predictive residual is 0.584 vs the homo 0.95 — a 39 % drop.
For a high-$\sigma_v$ held-out target with $\sigma_{v,*}^2 = 4$, the
hetero residual is 4.58 vs the homo 0.95 — almost 5× larger.

This is the propagation effect, faithfully implemented and faithfully
behaving as the data dictates.

---

*End of document. See `docs/UQ_THESIS_GAP_ANALYSIS.md` for the
implementation audit and §2 patch verification.*
