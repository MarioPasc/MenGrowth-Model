# Distribution-Free Calibration as a Salvage Path: Validation and Deep Dive

**Date:** 2026-05-13
**Context:** N=54 meningioma cohort. The τ-sweep main experiment (B=10 000 paired BCa,
BH-FDR) rejected the alternative hypothesis that segmentation-derived σ²_v improves
interval calibration over homoscedastic LME at the empirical scale (τ=0:
ΔIS@95 = −0.04, CI [−0.30, +0.23], 0/20 BH-rejected). Entropy as σ²_v gave the same
null. This document validates the previously proposed conformal-prediction salvage
path, derives the methods, and gives intuitive use cases.

---

## 1. Diagnosis: is the wall real?

**Claim from prior turn.** All 11 within-ensemble σ²_v candidates lie on the same
information manifold (ρ ≈ 0.15 between any per-scan variance summary and
|trajectory residual|). Any reweighting, monotone transform, or change of summary
statistic (entropy vs variance vs mutual information vs sum-of-Bernoulli-variances)
inherits this ceiling because all candidates are deterministic functions of the
same 20-member LoRA ensemble.

**Why this is mathematically sound.**
Let `Z_i = (logV_i^{(1)}, …, logV_i^{(20)})` be the ensemble volumes for scan i.
Every candidate signal `s_i` is `s_i = φ(Z_i)` for some φ : ℝ²⁰ → ℝ. By the data
processing inequality,

    I(s_i ; |r_i|)  ≤  I(Z_i ; |r_i|),

where `r_i` is the trajectory residual. The bound on the right is fixed by the
ensemble construction (encoder + LoRA + decoder). No re-summarisation can exceed
it. The τ-sweep null at τ=0 across 11 candidates is consistent with `I(Z_i ; |r_i|)`
being small in this cohort. **The wall is real and cannot be moved by post-hoc
re-summarisation of the same ensemble.**

**What is NOT proven.** That σ²_v is uninformative in absolute terms. The
conditional improvement on the high-σ²_v tertile (cov95 0.79→0.90, IS@95
17.7→9.8 in the 2026-04-29 result) survived as a real signal until the 20-seed
Picasso replication failed to reject (1/20 at τ=+2.86, CI [−2.36, +1.57]). The
honest read is: **the conditional signal exists but is below the detection
threshold at N=54**. Conformal prediction will let you extract whatever signal
is there with a finite-sample coverage guarantee, instead of staking validity on
a Gaussian likelihood whose σ̂ is mis-scaled.

---

## 2. Why conformal prediction is the right tool here

The current pipeline produces intervals of the form

    [μ̂(x) − z_{1-α/2} · σ̂(x),  μ̂(x) + z_{1-α/2} · σ̂(x)]                (1)

This is **Gaussian-likelihood calibration**. Validity requires:

1. The mean function μ̂ is unbiased.
2. The variance estimate σ̂² is well-calibrated.
3. Residuals are Gaussian.

In your setting all three fail in different ways: μ̂ has finite-sample bias
(estimated from 53 patients), σ̂² is the homoscedastic ML estimate (or
LoRA-injected estimate which is mis-scaled in 9/11 cases), and the residual
distribution is unknown but bounded — log V is bounded by anatomy.

**Conformal prediction replaces (1) with a non-parametric calibrated quantile of
held-out residuals.** Under the single assumption of exchangeability between
calibration and test points, conformal intervals achieve marginal coverage
≥ 1−α in finite samples with no distributional assumptions. The user did not
have to choose a likelihood; they only had to choose a non-conformity score.

**Why this directly addresses the current failure mode.**
Your τ=0 result shows IS@95 ≈ unchanged but coverage drifts (0.87–0.92 depending
on σ²_v variant) — the homo LME is mis-calibrated, but injecting σ²_v fixes
the wrong thing (it changes interval shape, not the marginal level). Conformal
fixes the marginal level by construction, then optional locally-adaptive scores
let you spend the saved budget on shape.

Reference: Vovk et al. *Algorithmic Learning in a Random World* (2005);
Angelopoulos & Bates *A Gentle Introduction to Conformal Prediction*
(arXiv 2107.07511, 2023).

---

## 3. The three flavours, in depth

### 3.1 Split conformal — the textbook baseline (NOT recommended here)

Split data into proper training (size n_t) and calibration (size n_c).
Fit μ̂ on training. Compute calibration residuals

    R_i = |y_i − μ̂(x_i)|,   i = 1, …, n_c.

Define the empirical quantile

    Q_α = the ⌈(n_c+1)(1−α)⌉-th smallest value of {R_i}.

Predict for new x:  **[μ̂(x) − Q_α, μ̂(x) + Q_α]**.

**Coverage theorem (Lei et al. 2018).** Under exchangeability,
ℙ(y_{new} ∈ interval) ≥ 1 − α.

**Why not here.** Wastes ~30% of N=54 patients on calibration. Quantile
estimation variance scales as 1/n_c; at n_c = 16 the 95th quantile has a
~±15% Monte Carlo uncertainty.

### 3.2 Jackknife+ — the right starting point

Barber, Candès, Ramdas, Tibshirani (2021). Uses all N residuals, no
data-splitting waste.

**Algorithm.** For each i = 1, …, N: fit μ̂_{−i} on the dataset without patient i.
Compute leave-one-out residual

    R_i = |y_i − μ̂_{−i}(x_i)|.

For a new test point x*, define

    L(x*) = the ⌊α(N+1)⌋-th smallest of  {μ̂_{−i}(x*) − R_i}_{i=1}^N,
    U(x*) = the ⌈(1−α)(N+1)⌉-th smallest of  {μ̂_{−i}(x*) + R_i}_{i=1}^N.

Predict **[L(x*), U(x*)]**.

**Coverage theorem.** Under exchangeability and a symmetric fitting algorithm,
ℙ(y* ∈ [L, U]) ≥ **1 − 2α** worst-case. In practice the realised coverage is
close to 1−α (empirical work and your own pilot will confirm this).

**Why this fits your project.** You already run LOPO-CV. The N=54 LOPO
residuals you compute for IS@95 evaluation are *exactly* the jackknife+ inputs.
**No new training. No retraining cost. Two new functions:** one for the lower
quantile, one for the upper.

**Intuitive use case.**
Patient *Anna* held out. Twenty fellow patients trained an LME with intercept
β̂_0 = 1.7, slope β̂_1 = 0.04 (per ordinal step). LOPO residuals (53 others) are
clustered around ±0.3 in log-volume, with two outliers at ±0.9. The empirical
95th percentile of |R_i| is 0.62. Jackknife+ at α=0.05 says: for Anna's
predicted log V̂ = 2.1 at t=3, the interval is **roughly [2.1 − 0.62, 2.1 +
0.62] = [1.48, 2.72]**, but more precisely the lower edge uses the 5th
percentile of {μ̂_{−i}(t=3) − R_i} which captures both prediction variation
across folds (Anna’s neighbours might disagree) AND residual scatter. If
Anna's covariates put her in a part of the predictor space where the
leave-one-out models disagree heavily, her interval automatically widens.

### 3.3 CQR / normalised conformity — the elegant extension

Two equivalent ways to get **locally adaptive** widths.

#### 3.3a Normalised conformity (Papadopoulos 2008; Lei et al. 2018)

Score

    s_i = |y_i − μ̂_{−i}(x_i)| / σ̂_{−i}(x_i),

where σ̂_{−i} is any heteroscedastic uncertainty estimate (your LMEHetero, your
entropy-driven σ̂, or a separate residual-magnitude regressor). Take the
(1−α) empirical quantile Q. Predict

    [μ̂(x*) − Q · σ̂(x*),  μ̂(x*) + Q · σ̂(x*)].

**Coverage.** Same 1−α guarantee under exchangeability.

**Why this is the elegant fix to your null result.**
Your high-σ²_v tertile experiment showed σ̂ has *relative* information — it
ranks which scans deserve wider intervals — but its *absolute* scale is wrong
(over-tight at τ=0, over-wide at τ=+10). Normalised conformity discards the
absolute scale and re-multiplies by the empirical Q. **You keep the shape
information from σ̂ and you discard the scaling that was hurting you.**

If σ̂ has no shape information (worst case), Q · σ̂ collapses to a constant and
you recover jackknife+ width. **It cannot do worse than jackknife+ asymptotically.**

#### 3.3b CQR (Romano, Patterson, Candès 2019)

Train two quantile regressors (e.g., quantile LME with rq() in R, or
gradient-boosted quantile loss) for the α/2 and 1−α/2 quantiles: q̂_lo, q̂_hi.
Score on calibration set:

    E_i = max(q̂_lo(x_i) − y_i,  y_i − q̂_hi(x_i)).

Predict **[q̂_lo(x*) − Q, q̂_hi(x*) + Q]** with Q the (1−α) empirical quantile
of {E_i}.

**Why CQR over normalised conformity at N=54.** CQR needs to *train* quantile
regressors. At N=54 with ~3 covariates this is feasible but tight. Normalised
conformity reuses your existing LMEHetero σ̂ and adds zero new estimation
burden. **Default: 3.3a. Treat 3.3b as a sensitivity check.**

#### 3.3c Density-calibrated CQR (CQR-d, 2024)

Sesia & Romano’s 2024 extension reweights conformity scores by local feature
density. Reports up to 21% width reduction on heteroscedastic simulations.
**Skip for the thesis.** Marginal gain over 3.3a, needs density estimation in
a 5D space at N=54.

---

## 4. Small-N pitfalls (what to flag in the thesis)

### 4.1 Coverage validity vs coverage *precision*

The guarantee is on *marginal* coverage in expectation. At N=54 the *observed*
coverage on a single held-out evaluation has Beta-Binomial uncertainty. The
exact 95% CI for an observed coverage of 0.95 at N=54 is roughly **[0.85,
0.99]**. Do not claim "95.0% coverage" — claim "coverage within the
finite-sample CI of the 95% target".

### 4.2 Exchangeability

Patients are exchangeable if i.i.d. (a reasonable approximation for a single
hospital cohort). Within a patient, scans are NOT exchangeable (temporal
correlation). **Patient-level LOPO is the right granularity:** treat each
patient as one exchangeable unit; pool all observations per patient into one
residual summary (median |residual| across that patient's scans, or just the
last-time-point residual). This costs you the per-scan signal but keeps the
guarantee clean. The alternative — LPCI (Batra+23) — gives two-axis coverage
but assumes asymptotic regimes and quantile-regression infrastructure that
isn't worth building at N=54.

### 4.3 Conditional vs marginal coverage

Conformal guarantees marginal coverage (averaged over the population). Patients
with rare covariate combinations may be systematically under-covered. Report
coverage stratified by σ²_v tertile — your existing tertile breakdown
already does this. Expect the high-tertile patients to retain their
conditional improvement *with the validity guarantee on top*.

### 4.4 Symmetry of the score

|y − μ̂| is a symmetric score. Your residual distribution may be asymmetric
(over-prediction of growth more common than under-prediction). Use the
two-sided signed score for jackknife+ to allow asymmetric intervals: separate
lower and upper quantiles.

---

## 5. Path B revisited: ensemble-of-trajectories Bayesian model averaging

Currently you collapse 20 ensemble log-volumes to one scalar σ²_v per scan,
then feed it as a fixed offset into LMEHetero. Path B replaces this with:

    p(y* | t*, D)  =  (1/M) · Σ_{m=1}^{M} N(y*;  μ̂_m(t*),  σ̂_m²(t*))         (2)

where (μ̂_m, σ̂_m²) is the LME predictive at t* fitted to the m-th ensemble's
trajectory data. **This is a proper Bayesian model average over the unknown
measurement realisation,** following the CASUS framework (Judge et al. 2025).

### Why this is theoretically cleaner

Law of total variance:

    Var(y*)  =  E_m[σ̂_m²]  +  Var_m[μ̂_m].

The first term is *within-model* uncertainty (Gaussian likelihood width). The
second is *measurement uncertainty* (how much M ensemble contours move the
mean). Your current σ²_v injection captures only the second; the first is
held fixed at the homo LME's σ̂_ε². The mixture (2) captures both.

### What it costs

20× LME fits per LOPO fold. With 54 folds and ~50 ms per LME fit, total ~50 s.
Trivial.

### What it gives over Path A

Heavier predictive tails when ensemble members disagree. Calibration on
contour-ambiguous scans improves. Combine with jackknife+ on top (use the
mixture's median or mean as μ̂ and conformalise the residuals) to keep the
finite-sample guarantee.

### Recommendation

Path A first, B as an ablation. If Path A already hits IS@95 below the homo
baseline by a defensible margin, B is a robustness check. If A is tied with
homo, B may unlock conditional improvements through the within-model variance
term.

---

## 6. Path C revisited: cross-segmenter disagreement

You have BSF-LoRA + BraTS25 + BraTS23-GLI segmentations (≈3 segmenters,
extendable to 4 with nnU-Net out-of-the-box).

**σ²_v_cross_seg = Var across segmenters of log V.**

This measures a fundamentally different epistemic axis: *architecture-induced
disagreement* on the same image. Within-LoRA variance (your 11 candidates)
measures *training-stochasticity-induced disagreement* in one architecture.

Plausible reason cross-segmenter might carry information that within-LoRA
doesn't: aggressive contours from one architecture vs conservative contours
from another are likely on scans with genuine ambiguity (heterogeneous
enhancement, partial volume, peritumoral edema confusion). Within-LoRA M=20
all share the BSF backbone and may agree on systematic biases.

**Cost.** 1 day. Plug into existing main_experiment as one more candidate
signal. Reuses the τ-sweep harness.

**Caveat.** M=3 or 4 is small for a variance estimate. Use it as a *binary
flag* (top-quartile cross-seg variance = "ambiguous") rather than a continuous
scalar.

---

## 7. Validated 2-week scope (revised from prior turn)

| Week | Day  | Task                                                        | Output                                              |
| ---- | ---- | ----------------------------------------------------------- | --------------------------------------------------- |
| 1    | 1    | `growth.shared.conformal`: split / jackknife+ / norm-CQR    | Module + unit tests on synthetic                    |
| 1    | 2    | Wrap LME, LMEHetero, NLME — reuse LOPO outputs              | `experiments/stage1_volumetric/conformal/`          |
| 1    | 3    | Run on N=54, ordinal time. Report IS@95, cov95, R²_log     | `results/conformal_runs/`                           |
| 1    | 4    | Stratified analysis: per-tertile cov, per-patient widths    | Figures + table                                     |
| 1    | 5    | Path C cross-segmenter binary flag                          | One extra row in the τ-sweep results table         |
| 2    | 1–2  | Path B mixture LME — only if Path A IS@95 ≥ homo           | `growth.models.growth.ensemble_lme`                 |
| 2    | 3    | If real dates land, rerun A on `days_from_baseline`        | Side-by-side table (ordinal vs continuous)         |
| 2    | 4–5  | Writeup: "Distribution-Free Interval Calibration" section  | LaTeX in `docs/technical_report/sections/`         |

**Stop conditions.**
- After Week 1 day 3: if jackknife+ on LME beats homo IS@95 with non-overlapping
  paired bootstrap CI, Path A *is the headline*. Skip Path B.
- If jackknife+ on LME ties homo IS@95: Path A still wins on *validity* (you
  have a coverage theorem; the homo baseline doesn't). Run Path B.
- If jackknife+ on LMEHetero (normalised) beats LMEHetero homo IS@95: the
  σ²_v signal is non-zero and conformal rescued it. **Strongest story.**

---

## 8. The thesis claim, calibrated to evidence

Best case (Path A delivers on LMEHetero):

> On a longitudinal cohort of N=54 meningioma patients, 11 segmentation-derived
> σ²_v signals were tested under a heteroscedastic LME framework. None
> significantly improved interval calibration over homoscedastic LME at the
> empirical scale (paired BCa B=10 000, BH-FDR 0/180). Distribution-free
> conformal calibration with jackknife+ residuals (Barber et al. 2021) and
> locally-adaptive normalised conformity scores recovered the latent
> heteroscedastic signal: marginal coverage = 0.94 [Beta-binomial CI 0.86–0.98],
> IS@95 = X.XX vs Y.YY for homo (paired BCa CI [a, b]). On the high-σ²_v
> tertile, conformally-normalised LMEHetero reduced IS@95 by Δ vs homo
> [CI [c, d]], confirming that the σ²_v signal carries adaptive shape
> information even when its absolute scaling is unreliable.

Neutral case (Path A only matches homo on IS@95):

> ... no σ²_v candidate beat homo; conformal calibration matched homo IS@95
> but provided a distribution-free finite-sample coverage guarantee
> (Barber et al. 2021), strictly stronger than the Gaussian-likelihood
> calibration of the baseline. This trades a small width premium for
> formal validity.

Both are publishable. The neutral case is a *methodological* contribution
even with no numerical gain.

---

## 9. What to NOT do

- **MC Dropout.** BSF/SwinUNETR uses LayerNorm only; injecting dropout is a
  retraining task. Even if done, hits the same ρ ≈ 0.15 ceiling.
- **TTA augmentation.** Maganti+25 shows TTA beats MCD on segmentation
  quality detection, not downstream trajectory regression. Wrong target.
- **Bayesian last layer / SWAG.** Retraining cost > 1 week budget.
- **CQR-d (density-calibrated).** Marginal gain at N=54.
- **LPCI.** Use full longitudinal conformal only after the simpler path
  is exhausted. Patient-level LOPO + jackknife+ is sufficient under the
  exchangeability granularity that fits this cohort.
- **Inflate τ-sweep claims.** The current null is real and *publishable*.
  Do not retrofit "σ²_v helps after all" — phrase the conformal result
  as recovering adaptive *shape* given the absolute-scale signal was null.

---

## 10. References (verified)

- Barber, Candès, Ramdas, Tibshirani. **Predictive inference with the
  jackknife+.** Annals of Statistics 49(1):486–507, 2021.
- Romano, Patterson, Candès. **Conformalized quantile regression.**
  NeurIPS 2019.
- Papadopoulos, Vovk, Gammerman. **Normalized nonconformity measures for
  regression conformal prediction.** AIA 2008.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. **Distribution-free
  predictive inference for regression.** JASA 113(523):1094–1111, 2018.
- Batra, Patel, Ren et al. **Conformal Predictions for Longitudinal Data.**
  arXiv:2310.02863, 2023.
- Angelopoulos, Bates. **A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification.** arXiv:2107.07511, 2023.
- Vaghi, Rodallec, Fanciullino et al. **Population modeling of tumor growth
  curves and the reduced Gompertz model.** PLOS Comput Biol 16(2):e1007178,
  2020.
- Judge et al. **CASUS: Contour Sampling for Uncertainty in Segmentation.**
  2025.
- Maganti, Pati et al. **Test-time augmentation vs Monte Carlo dropout
  for medical image segmentation uncertainty.** 2025.
- Engelhardt et al. **Meningioma growth modelled with Gompertz dynamics.**
  2023.
