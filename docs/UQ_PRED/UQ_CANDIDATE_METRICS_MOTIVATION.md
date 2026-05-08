# Why Test Alternative Segmentation-Uncertainty Metrics?

*Companion to `experiments/stage1_volumetric/test_candidate_uncertainty_signals/`.
Pre-registers the diagnostic and lays out the empirical evidence that motivates it.*

---

## 1. Stage 1 question

The Stage 1 chapter of the thesis evaluates whether propagating per-scan
segmentation uncertainty $\sigma^2_v$ from a LoRA ensemble into a heteroscedastic
LME (LMEHetero) tightens the predictive interval at calibration-relevant
patients. The current $\sigma^2_v$ is the variance of log-volume across the
$M=20$ members,

$$
\sigma^2_{v,k} \;=\; \mathrm{Var}_{m=1..M}\!\left[\log\!\bigl(V_k^{(m)} + 1\bigr)\right].
$$

The main experiment, the τ-shift sweep, and the synthetic stress test all
report a marginally null effect. Two competing explanations remain open:

* **H1 — wrong summary statistic.** The scalar $\sigma^2_v$ projects spatial
  ensemble disagreement onto a single number and discards boundary, region,
  and shape information that *might* track the trajectory residual.
* **H2 — information-poor channel.** The segmentation channel is, at any
  aggregation level, uncorrelated with biological growth volatility — the
  dominant source of residual on the meningioma cohort.

This document collects the evidence that makes both hypotheses live, and
specifies the candidate metric set that distinguishes them.

---

## 2. Empirical record so far

### 2.1 Main experiment — τ=0 (empirical) is statistically null

Paired BCa bootstrap, $B=10\,000$, BH-FDR across 20 seeds,
N=54 patients, LOPO-CV, last-from-rest:

| τ value | $\Delta\mathrm{IS}@95$ vs LME-homo | 95 % CI | BH-rejected (q=0.05) |
|---|---:|---|---:|
| $0$ (empirical) | $-0.04$ | $[-0.30,\, +0.23]$ | $0/20$ |
| $+2.86$ (point min) | $-0.36$ | $[-2.36,\, +1.57]$ | $1/20$ |
| $\geq +10.33$ | $\geq +11$ | wide, all positive | $20/20$ |

No empirical scale of $\sigma^2_v$ improves IS at any patient-level paired
comparison. Hetero LME is "safe but unhelpful": injecting empirical $\sigma^2_v$
neither degrades nor improves calibration over the homoscedastic baseline
(`docs/UQ_PRED/UQ_THESIS_GAP_ANALYSIS.md`).

### 2.2 Synthetic stress test — most of the homo–hetero gap is structural

A controlled $\sigma^2_v$ injection over four shape profiles (uniform-zero,
constant-non-zero, oracle-correlated, and τ-swept empirical) attributes the
LME→LMEHetero score difference as

* $\sim 91\%$ structural — the extra variance term in LMEHetero's predictive
  distribution that exists even when $\sigma^2_v$ carries no information,
* $\sim 9\%$ genuine $\sigma^2_v$ propagation gain.

Profile B (uninformative dispersion) is the negative control showing that the
gain only materialises when $\sigma^2_v$ is correlated with the residual; the
empirical $\sigma^2_v$ on this cohort is closer to profile B than to the oracle
(`docs/UQ_PRED/UQ_SYNTHETIC_VARIANCE_STRESSTEST_RESULTS.md`).

### 2.3 IS decomposition — width and miss are tied across tertiles

Per-fold decomposition $\mathrm{IS}_\alpha = (U-L) + (2/\alpha)\bigl[(L-y)_+ + (y-U)_+\bigr]$
(figure at `results/.../is_decomposition_homo_vs_hetero.pdf`):

| tertile of held-out σ²_v | $n$ | $\bar\sigma^2_v$ | homo IS / W / M | hetero IS / W / M |
|---|---:|---:|---|---|
| low | 18 | 0.014 | 4.76 / 4.41 / 0.35 | 4.74 / 4.31 / 0.43 |
| mid | 18 | 0.030 | 11.08 / 4.39 / 6.68 | 11.12 / 4.32 / 6.81 |
| high | 18 | 0.066 | 9.02 / 4.45 / 4.57 | 8.88 / 4.42 / 4.47 |
| marginal | 54 | 0.037 | **8.29** / 4.42 / 3.87 | **8.25** / 4.35 / 3.90 |

Width contracts by $\sim 2\%$ from homo to hetero in every tertile and miss
moves by $\le 0.15$ in either direction. The "same IS, different routes"
trade-off is real but quantitatively tiny: $\hat\sigma_n \approx 1.13$
(implied by mean width $4.4$ in log-volume units), so $\sigma^2_v \in [0.014, 0.066]$
is one to two orders of magnitude below the LME's residual variance.

### 2.4 The τ-sweep cannot manufacture the missing information

A uniform log-shift $\sigma^2_{v}(\tau) = \exp(\tau)\cdot\sigma^2_{v,\mathrm{emp}}$
preserves the *shape* of the empirical distribution and only inflates
*scale*. Because the miss penalty has a floor at zero (once covered, no further
gain) but width grows monotonically with $\sigma$, the IS surface
saturates: at large τ all observations are covered and $\mathrm{IS} = U - L$
grows as $\sqrt{\exp(\tau)}$. There is no τ for which the empirical $\sigma^2_v$
distribution would beat homo, because no monotone scaling of an information-poor
ranking adds information.

### 2.5 The diagnostic that the prior evidence asks for

The Stage 1 LME residual at the held-out timepoint decomposes as

$$
y_* - \hat\mu_*^{\mathrm{homo}}
\;=\; \underbrace{(\hat V_* - V_*^{\mathrm{true}})/V_*^{\mathrm{true}}}_{\text{segmentation}}
\;+\; \underbrace{\delta_{\mathrm{Gompertz}}(t_*)}_{\text{first-order linearisation}}
\;+\; \underbrace{\eta_{\mathrm{biology}}(t_*)}_{\text{growth volatility}}.
$$

For LMEHetero to beat LMEHomo, $\sigma^2_{v,*}$ must correlate with
$|y_* - \hat\mu_*^{\mathrm{homo}}|$. The current scalar $\sigma^2_v$ measures the
first term only and projects it onto one number; if the second and third terms
dominate, even a perfect segmentation-side estimator will not move IS on this
cohort. Stage 1 of the diagnostic computes this correlation directly.

---

## 3. Candidate metrics

Each candidate is a per-scan scalar derived from the M=20 LoRA ensemble.
Mathematical formulae assume $C$ output channels (BraTS-TC, WT, ET) of
sigmoid probability $p_{m,v} \in [0,1]$ for member $m$ and voxel $v$.

| code name | family | formula | intuition |
|---|---|---|---|
| `logvol_var` (baseline) | epistemic, scalar | $\mathrm{Var}_m[\log(V^{(m)}+1)]$ | current main-experiment $\sigma^2_v$ |
| `logvol_mad_var` | epistemic, robust | $(1.4826\cdot\mathrm{MAD}_m)^2$ | outlier-robust analogue of `logvol_var` |
| `vol_cv2` | relative | $(\sigma_V / \mu_V)^2$ | dimensionless segmentation noise |
| `mean_entropy` | total, voxel-avg | $\frac{1}{|\Omega|}\sum_v H[\bar p_v]$ | total predictive uncertainty (Kendall–Gal 2017) |
| `mean_mi` | epistemic (BALD) | $\frac{1}{|\Omega|}\sum_v \bigl(H[\bar p_v] - \overline{H[p_{m,v}]}\bigr)$ | epistemic part — what would shrink with more members (Houlsby 2011, Gal 2017) |
| `mean_var_voxel` | epistemic, voxel-avg | $\frac{1}{|\Omega|}\sum_v \mathrm{Var}_m[p_{m,v}]$ | voxelwise ensemble dispersion |
| `men_entropy` | total, MEN-restricted | $H[\bar p_v]$ averaged over `mean_p[0] > 0.5 ∩ CC≥64` | total entropy where the tumour actually is |
| `men_mi` | epistemic, MEN-restricted | BALD averaged over MEN region | epistemic on tumour mass |
| `men_boundary_entropy` | total, boundary band | $H[\bar p_v]$ averaged over (dilated MEN \ MEN) | aleatoric proxy: ambiguity at the *contour*, where segmentation actually fails |
| `men_boundary_mi` | epistemic, boundary | BALD on the boundary band | epistemic on the ambiguous rim |
| `composite` | composite | $\sigma^2_{\log V}\cdot(1 + \beta H_{\mathrm{boundary, MEN}})$ | volume noise *modulated* by boundary ambiguity |

**Negative controls** (mandatory for protocol validity):

* `zero` — $\sigma^2_v \equiv 0$. LMEHetero ≡ LMEHomo (pipeline check).
* `constant_mean` — $\sigma^2_v \equiv \overline{\sigma^2_v}_{\mathrm{emp}}$.
  Homo with shifted $\sigma_n$.
* `permuted` — random shuffle of empirical $\sigma^2_v$ across scans.
  Tests whether shape information matters at all.

Each candidate is run under two scalings (`raw`, `mean_matched`) so that we
disentangle absolute scale from rank/shape information.

---

## 4. Pre-registered hypotheses

### Stage 1 — information content

For each candidate $c$, compute Spearman $\rho$, Pearson $r$, and Kendall $\tau$
between $c_k$ and $|y_k - \hat\mu_k^{\mathrm{homo}}|$ across the 54 LOPO
held-out predictions. 95 % BCa bootstrap CI ($B=10\,000$).

* **Pass threshold (informative)**: any candidate with $|{\hat\rho}| > 0.20$
  whose 95 % CI excludes zero. This is the practical-significance bar
  Steiger 1980 adopts for "weak but real" rank correlations.
* **Reject threshold (information-poor)**: every candidate with
  $|{\hat\rho}| < 0.20$ across all three correlation estimators. In that
  regime no monotone transform of any candidate will move IS at this $N$.

### Stage 2 — downstream IS

For each (candidate, scaling) cell run LOPO-LMEHetero and compute paired
$\Delta\mathrm{IS}@95$ vs LME-homo with $B=10\,000$ patient-level BCa
bootstrap. Apply BH-FDR ($q=0.05$) across all cells.

* **Pass**: at least one cell has $\Delta\mathrm{IS}@95 < 0$ with
  BH-rejected null. Hypothesis H1 wins; the propagation pipeline is
  salvageable with the right summary statistic.
* **Reject**: no cell has BH-rejected $\Delta\mathrm{IS}@95 < 0$.
  Hypothesis H2 wins; the segmentation channel is information-poor for
  trajectory residuals on this cohort.

### Joint outcome map

| Stage 1 outcome | Stage 2 outcome | Thesis claim |
|---|---|---|
| any $|\rho|>0.20$ | any cell BH-significant | "Candidate $X$ is the right propagation signal." Replace `logvol_var` in the methodology chapter. |
| any $|\rho|>0.20$ | no cell significant | The signal exists but is too weak to overcome LMEHetero's structural cost at $N=54$. Argue the bound. |
| all $|\rho|<0.20$ | (any) | Strong negative result — segmentation uncertainty does not predict trajectory residuals at any aggregation. Future work should look outside the LoRA ensemble (test-retest variance, trajectory-residual bootstrap). |

---

## 5. Why this is the right experiment to run before the thesis hand-in

1. **It separates the structural cost from the missing information.** The
   τ-sweep showed scale doesn't help; the decomposition figure showed
   width and miss are tied; the synthetic stress test attributed 91 %
   of the score gap to the structural variance term. A negative result
   on Stage 1 (no candidate exceeds the rank-correlation threshold) is
   the cleanest closure of that line of evidence.

2. **It is cheap.** Stage 1 is a one-shot correlation against an existing
   `LME_baseline/lopo_results.json`; it runs in seconds. Stage 2 is 26
   LMEHetero LOPO runs (≈ 5 min each on one A100, ≈ 25 min total on 8
   concurrent), reusing every line of `main_experiment` infrastructure.

3. **It preserves narrative continuity.** The thesis already commits to
   "propagation of segmentation uncertainty through LMEHetero". The
   diagnostic does not change that frame — it asks which *summary* of the
   ensemble carries the propagation, and reports the answer with the
   same statistical machinery (paired BCa, BH-FDR, tertile stratification)
   that the main experiment uses. Whichever way Stage 2 lands, the result
   is publishable evidence either *for* a specific replacement statistic
   or *against* the segmentation channel as the source of trajectory
   uncertainty on small longitudinal cohorts.

---

## 6. References

* Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation*. JASA 102:359 — IS as the headline metric.
* Gneiting, T. & Katzfuss, M. (2014). *Probabilistic Forecasting*. Annu.
  Rev. Stat. — sharpness vs calibration trade-off.
* Kendall, A. & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian
  Deep Learning for Computer Vision?* NeurIPS — aleatoric / epistemic
  decomposition that motivates the boundary-entropy candidate.
* Houlsby, N. et al. (2011). *Bayesian Active Learning by Disagreement*. —
  BALD / mutual-information family.
* Gal, Y., Islam, R. & Ghahramani, Z. (2017). *Deep Bayesian Active
  Learning with Image Data*. ICML — BALD applied to deep ensembles.
* Wickstrøm, K., Kampffmeyer, M. & Jenssen, R. (2020). *Uncertainty and
  Interpretability in Convolutional Neural Networks for Semantic Segmentation*.
  Med. Image Anal. 60:101619 — boundary-targeted uncertainty.
* Mehrtash, A. et al. (2020). *Confidence Calibration and Predictive
  Uncertainty Estimation for Deep Medical Image Segmentation*. IEEE TMI
  39:3868 — calibration of segmentation ensembles.
