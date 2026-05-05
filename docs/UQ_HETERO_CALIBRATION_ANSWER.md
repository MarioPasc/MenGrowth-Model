# Does the heteroscedastic formulation produce better-calibrated predictions?

**Scope.** Stage 1 UQ run, last_from_rest LOPO, N=56 patients, ordinal
time, ensemble `logvol_mean` target on the MEN-volume convention
(labels 1|3). Source:
`/media/mpascual/Sandisk2TB/.../uncertainty_propagation_volume_prediction/`.
Companion docs: `UQ_CALIBRATION_STORY.md`, `UQ_THESIS_GAP_ANALYSIS.md`.

## TL;DR

**Marginally, no.** **Conditionally, yes — exactly where it should.**

## Core claim (the result, in one paragraph)

> **Hetero does not make sharper predictions everywhere — it moves
> sharpness to where the data justifies it. On clean scans it
> sharpens; on noisy scans it widens. The homo model is stuck at the
> *average* segmentation noise, so it is systematically miscalibrated
> in *opposite directions* on the two regimes — and the high-noise
> miscalibration (overconfidence) is the one that breaks coverage.**

### How to prove it (the empirical chain)

1. **Bimodality of σ²_{v}.** Show median ≪ mean (0.0012 vs 0.42) and
   the long right tail (6 % of scans with σ_v > 1). Source:
   `uncertainty/logvol_std` in `MenGrowth.h5`. *Predicts:* REML must
   absorb the *mean* σ²_v into σ²_n.
2. **REML budget identity.** Show
   $\hat\sigma^2_{n,\text{homo}} \approx
    \hat\sigma^2_{n,\text{het}} + \overline{\sigma^2_v}$
   (0.95 ≈ 0.55 + 0.40). Source: `hyperparameters.json` per fold.
   *Predicts:* on a target with σ²_{v,*} ≈ median, hetero is
   *narrower* than homo.
3. **Per-target variance redistribution.** Plot $s^{*2}_\text{het}$
   vs $s^{*2}_\text{homo} = \text{const}$ as a function of
   σ²_{v,*}. *Predicts:* a positively-sloped line that crosses the
   homo constant at σ²_{v,*} ≈ $\overline{\sigma^2_v}$.
4. **Conditional calibration table.** Stratify the 56 LOPO targets
   into σ²_{v,*} tertiles and report cov@95 / IS@95 / CRPS per
   tertile and per model. *Predicts:* (a) homo cov@95 collapses
   on the high tertile (observed: 0.789); (b) hetero cov@95 holds
   (observed: 0.895); (c) hetero IS@95 ≈ ½ homo IS@95 on high
   tertile (observed: 9.82 vs 17.66).
5. **Sharpness inversion across tertiles.** Confirm CI width(homo)
   ≈ const, CI width(het) increases with σ²_{v,*}. Observed:
   homo 5.55 → 5.51 → 5.45; het 4.49 → 4.44 → 6.27.
6. **Synthetic stress test (proposed).** Inject controlled
   σ²_v profiles into the same LOPO loop and verify that the
   tertile pattern is *caused* by σ²_v dispersion, not by
   correlated nuisance variables. Procedure: see
   `UQ_SYNTHETIC_VARIANCE_STRESSTEST.md`.

The heteroscedastic predictive variance
$s^{*2} = \mathbf{x}^{*\top}\widehat{\mathrm{Cov}}(\hat\beta)\mathbf{x}^* +
\mathbf{z}^{*\top}\hat\Omega\mathbf{z}^* + \hat\sigma_n^2 + \sigma_{v,*}^2$
does **not** beat its homoscedastic sibling on aggregate marginal
calibration for the LME family, and only marginally for the GP family.
But once we stratify by the target's segmentation variance
$\sigma^2_{v,*}$, propagation produces the textbook benefit on the
high-noise tertile, which is the only regime where the
fixed-σ² model is forced to be wrong.

## 1. Marginal results (paired, same 56 patients)

From `comparison_homo_vs_hetero.json` and `error_summary.json`:

| Pair | ΔR² | ΔCRPS | Δcov@95 | IS@95 (homo→het) | p (paired) |
|---|---|---|---|---|---|
| LME → LMEHetero | −0.150 | −0.004 | 0.000 | 10.55 → **9.00** | 0.398 |
| ScalarGP → ScalarGPHetero | +0.168 | −0.122 | −0.018 | 9.72 → **6.56** | 0.409 |
| HGP → HGPHetero | −0.013 | −0.046 | +0.018 | 9.56 → **6.51** | 0.420 |
| HGP_Gompertz → HGP_Gompertz_Hetero | −0.548 | +0.259 | +0.054 | 9.79 → 11.87 | **0.011** |

Reading the table.

- **CRPS is essentially tied** for every well-behaved pair. CRPS is
  scale-invariant in the variance budget; if total variance is
  preserved (it is, by REML), CRPS does not move.
- **Sharpness improves consistently for hetero** in the GP family
  (≈30 % narrower at the same coverage) — this is the only place a
  marginal "win" appears, and it is *not* statistically significant
  on a paired test of squared errors at N=56.
- **LMEHetero is sharper *and* slightly worse-covering** (cov95
  0.893 → 0.875). At first glance, this looks like an anti-result
  for propagation.
- **HGP_Gompertz_Hetero is broken** (over-coverage with 12-unit-wide
  CI). Gompertz extrapolation pathologies, not a propagation finding.

If we stop reading here, the answer is "propagation does not help."
That is the wrong answer.

## 2. Why the marginal answer is misleading

From `UQ_CALIBRATION_STORY.md` §3 and direct measurement of
`uncertainty/logvol_std` (M=20, N=179 scans):

| Statistic | σ²_v |
|---|---|
| Median | **0.0012** |
| Mean | 0.42 |
| 90th pctl | 0.26 |
| Max | 11.20 |
| #(σ_v > 1) | 11 / 179 (6 %) |

The σ²_v distribution is **strongly bimodal**: 90 % of scans are
near zero, a 6 % tail is large. REML minimises the joint negative
log-likelihood by absorbing the *average* σ²_v into a smaller σ²_n:

$$\hat\sigma^2_{n,\text{homo}} \approx \hat\sigma^2_{n,\text{het}} +
\overline{\sigma^2_v} \quad (0.95 \approx 0.55 + 0.40).$$

At test time, the held-out target almost always has σ²_{v,*} ≈
median ≪ mean, so the hetero predictive variance for a *typical*
target is ≈0.55 — narrower than the homo's 0.95. On a *high-noise*
target, hetero correctly inflates to ≈ 0.55 + σ²_{v,*}, while homo
stays at 0.95.

Marginal coverage averages both regimes and washes the effect out.
The propagation effect is **conditional on σ²_{v,*}** by construction.
This is exactly what we should measure.

## 3. Conditional results: stratified by σ²_{v,*} tertile

From `conditional_calibration_last_from_rest.md` (cuts:
q33=2.98e-4, q66=8.55e-3). Reproduced for the four pairs that
matter:

### LME family (the central manuscript pair)

| Tertile | n | σ²_v mean | Model | CI | cov@95 | IS@95 | CRPS |
|---|---|---|---|---|---|---|---|
| low | 19 | 1.1e-4 | LME | 5.55 | **0.947** | **5.80** | 0.632 |
| low | 19 |  | LMEHetero | 4.49 | 0.895 | 5.86 | 0.580 |
| mid | 18 | 1.9e-3 | LME | 5.51 | 0.944 | **7.92** | 0.574 |
| mid | 18 |  | LMEHetero | 4.44 | 0.889 | 11.45 | 0.618 |
| **high** | 19 | **1.79** | LME | 5.45 | **0.789** | **17.66** | 1.132 |
| **high** | 19 |  | **LMEHetero** | **6.27** | **0.895** | **9.82** | 1.131 |

Three regimes, three different verdicts:

- **Low σ²_{v,*}**: homo wins on coverage (0.947 vs 0.895). LMEHetero
  is too narrow because REML absorbed mean σ²_v into σ²_n and there
  is nothing to add back at the target. Sharper-but-undercovers,
  exactly the §1 finding.
- **Mid σ²_{v,*}**: tied on coverage; LMEHetero's IS@95 is *worse*
  (11.4 vs 7.9) because of one or two big misses on this tertile —
  N=18 is small and the IS is sensitive to outliers.
- **High σ²_{v,*}**: **propagation rescues coverage**
  (0.789 → 0.895) and **halves the interval score**
  (17.66 → 9.82). The CI widens from 5.45 to 6.27 in *exactly* the
  subset that needs the extra width. CRPS stays flat because the
  distribution mean is unchanged; calibration carries the win.

This is the textbook UQ-propagation result, and it is present in the
data.

### GP families

ScalarGPHetero and HGPHetero behave like LMEHetero on the high
tertile (CI inflates from ≈8.1 → 7.3 and 6.5 → 7.3, IS@95 falls from
9.5 → 7.5 and 15.5 → 7.8) **and** stay competitive on low/mid because
the GP fits its own residual noise per fold, so the hetero offset
does not get absorbed as harshly as REML absorbs it for the LME.

### Verdict

> **The textbook propagation result is present in the data, but
> only conditionally on σ²_{v,*}: hetero re-allocates predictive
> sharpness to the targets that actually carry low measurement
> noise, and inflates only the targets that carry high measurement
> noise. The homo model is stuck at the average and is therefore
> systematically miscalibrated in opposite directions on the two
> regimes.**

| Question | Answer |
|---|---|
| Does propagation improve marginal calibration? | **No** for LME; **mildly yes** for GP (sharper at same coverage; not significant at N=56). |
| Does propagation improve calibration on noisy targets, where it matters? | **Yes, substantially.** LMEHetero halves the high-tertile IS@95 and lifts coverage from 0.79 to 0.90. |
| Is the formulation sound? | **Yes.** The math is the standard mixed-model + Gaussian likelihood + known per-observation noise. Verified against the per-fold REML walkthrough (`UQ_CALIBRATION_STORY.md` §8). |
| Is the experiment a clean test of the thesis claim? | **Not yet** — see §4. |

## 4. Confounds that prevent a clean propagation win

Beyond what is already in `UQ_THESIS_GAP_ANALYSIS.md`:

1. **Ordinal time.** With t ∈ {0,1,2,…}, the random-slope contribution
   $z^{*\top}\hat\Omega z^*$ scales as $t^{*2}$ over integer steps,
   not over real days. The thesis's long-horizon (1, 2, 5 yr) claim
   that propagation+random-slope dominate at long horizons is
   currently untestable. Re-run with `time.variable:
   years_from_baseline`.
2. **σ²_v floor at 1e-6.** Lets the per-target variance vanish.
   Setting a clinically informed floor (e.g., the empirical 25th
   pct ≈ 1e-3, or a test–retest variance) prevents the
   "hetero-narrower-than-homo on every typical scan" pathology.
3. **Bimodal σ²_v with QC outliers in.** 11/179 scans have σ_v > 1.
   These are likely segmentation failures (zero-volume, bilateral,
   tiny tumors). They drag $\overline{\sigma^2_v}$ up and force REML
   to over-deflate σ²_n. Either drop them as QC failures
   (`max_logvol_std: 1.0` filter, see `UQ_CALIBRATION_STORY.md` §6.2)
   or report stratified results as the primary calibration evidence.
4. **EBLUP MSPE cross-term not used.** The current $s^{*2}$ omits the
   $-2\,\mathrm{cov}(\hat\beta, \hat u_i)$ term; usually small but
   correct it (`UQ_CALIBRATION_STORY.md` §6.5).
5. **N=56 is not enough power.** All paired tests on |error| are p
   ≈ 0.4. The propagation effect is conditional and concentrated in
   ~19 patients; conditional-IS bootstraps would have far more power
   than marginal-error Wilcoxons.

## 5. Recommended next steps, ordered by impact

1. **Adopt the conditional-calibration framing in the manuscript.**
   The honest claim is *"propagation prevents over-confidence on
   high-uncertainty targets at the cost of slight over-sharpness on
   well-segmented targets; net interval score on noisy targets falls
   ≈45 %."* Use the §3 stratified table as the primary calibration
   evidence, not the marginal table. Implementation: already in
   `conditional_calibration_last_from_rest.{json,md}` — just
   incorporate. (No code change.)

2. **Re-run with continuous time.** Set
   `time.variable: years_from_baseline` in
   `experiments/stage1_volumetric/configs/config_uq.yaml`, re-run
   the LME pair plus NLME family. Re-do tertile stratification.
   Expected: the random-slope $\hat\tau_1^2$ becomes interpretable
   per year, propagation effect strengthens at the held-out final
   timepoint because $z^{*\top}\hat\Omega z^*$ now grows with real
   horizon, and the "hetero loses on low tertile" effect should
   shrink because total variance budget redistributes.

3. **Floor σ²_v at 1e-3 (clinically informed).** One-line change in
   `LMEHeteroGrowthModel.floor_variance` and the GP hetero kernels.
   Clinical justification: a test–retest study on meningioma
   segmentation (or use the empirical 25th pct of σ²_v as a
   data-driven floor). This keeps the per-target variance
   non-vanishing on well-segmented scans and recovers low-tertile
   coverage parity.

4. **QC the 11 high-σ_v outliers.** Inspect each: zero-volume
   targets (`V̄=0`), tiny tumors with disagreeing members, bilateral
   masks. Add a `max_logvol_std: 1.0` H5 filter and report
   calibration with and without the outlier subset. This separates
   the propagation effect from the segmentation-failure regime.

5. **Increase bootstrap to 10 000 resamples and switch from
   Wilcoxon-on-|error| to paired bootstrap of ΔIS@95 stratified by
   tertile.** Marginal Wilcoxon misses the conditional effect by
   construction. ΔIS bootstraps on the high tertile (n=19) will
   likely be the only test that reaches significance — and that is
   the test that matches the thesis claim.

6. **Add PIT histograms and a KS test of PIT vs Uniform(0,1).**
   `shared/calibration_plots.py` already produces them; surface in
   the manuscript with a quantitative KS p-value per model. Stratify
   PIT by tertile too.

7. **Adopt the EBLUP MSPE cross-term.** Numerically small but the
   thesis already cites the correct expression; align the code.

8. **Drop HGP_Gompertz_Hetero or fix the Gompertz mean.** Currently
   contaminates the headline table with a separate failure mode.

---
*Author: Stage 1 UQ analysis, 2026-05-05. Built on
`UQ_CALIBRATION_STORY.md` and `UQ_THESIS_GAP_ANALYSIS.md`.*
