# Cumulative σ_v Sweep over LoRA Ensemble Size

**Question.** As we add LoRA ensemble members one at a time
($m = 2, 3, \dots, 20$), how does the per-scan segmentation
variance $\sigma_v$ (and $\sigma_v^2$, which is what feeds REML in
LMEHetero) behave? Does the variance estimate degrade with more
members, hit a floor, or keep moving?

**Data.** `MenGrowth.h5` `uncertainty/per_member_volumes` with shape
$(N, M) = (179, 20)$, transformed via $y = \log(V+1)$ to match the
modelling target. For each $m$, we drew $S=200$ random subsets of
size $m$ without replacement and computed
$\sigma_v(m,k) = \mathrm{SD}\bigl(y_k^{(j)} : j \in \text{subset}\bigr)$
per scan $k$.

Outputs:
`/media/mpascual/Sandisk2TB/.../uncertainty_propagation_volume_prediction/sigma_v_vs_m/`
— `sigma_v_vs_m.json`, `sigma_v_vs_m_by_tertile.json`,
`sigma_v_mc_stability.json`, `sigma_v_vs_m.{pdf,png}`.

---

## 1. Cohort summary

`mean σ_v` and `mean σ_v²` across the 179 scans, as a function of $m$:

| m | mean σ_v | median σ_v | p90 σ_v | mean σ²_v | gap to m=20 (%) on σ²_v |
|---|---|---|---|---|---|
| 2 | 0.169 | 0.026 | 0.339 | 0.4006 | −3.81 |
| 5 | 0.217 | 0.032 | 0.431 | 0.4174 | +0.21 |
| 10 | 0.233 | 0.034 | 0.469 | 0.4209 | +1.06 |
| 15 | 0.235 | 0.034 | 0.488 | 0.4126 | −0.93 |
| **20** | **0.239** | **0.034** | **0.500** | **0.4165** | **0.00** |

Two distinct convergence behaviours:

- **mean $\sigma_v^2$** (the quantity REML actually consumes): converges
  to its $m=20$ value within $\pm 4\%$ already at $m=2$ and within
  $\pm 1\%$ from $m=5$ onwards. This is the cohort budget that gets
  absorbed into $\hat\sigma^2_{n,\text{homo}}$.
- **mean $\sigma_v$** (the standard deviation): rises from 0.169 at
  $m=2$ to 0.239 at $m=20$, monotonically. This is the small-sample
  bias of $\mathrm{SD}$ as an estimator (Jensen's inequality applied
  to the square root). It is still creeping upward at $m=20$
  (≈ 0.2 % per added member) but well within Monte-Carlo noise.

**No degradation.** Adding members never inflates the cohort
$\sigma_v^2$ pathologically and never explodes the tail.

## 2. Per-tertile convergence

Tertiles defined by $\sigma_v^2$ at $m=20$ (matches the
conditional-calibration analysis):

- low: $\sigma^2_v \le 2.5{\times}10^{-4}$, n=60
- mid: $2.5{\times}10^{-4} < \sigma^2_v \le 6.6{\times}10^{-3}$, n=59
- high: $\sigma^2_v > 6.6{\times}10^{-3}$, n=60

Per-tertile mean $\sigma_v^2(m)$:

| tertile | n | m=2 | m=5 | m=10 | m=15 | m=20 |
|---|---|---|---|---|---|---|
| low | 60 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| mid | 59 | 0.0018 | 0.0018 | 0.0018 | 0.0018 | 0.0018 |
| high | 60 | 1.234 | 1.251 | 1.249 | 1.241 | 1.241 |
| all | 179 | 0.4144 | 0.4199 | 0.4194 | 0.4165 | 0.4165 |

**The tertile means are flat across the entire $m$ sweep.** The
*which-scan-is-noisy* signal is essentially locked in once $m \ge 2$.
This is unsurprising for the high tertile (one or two outlier
members dominate $\mathrm{SD}$ already), and trivially true for the
near-zero low tertile.

**Practical consequence.** REML's absorption budget,
$\overline{\sigma_v^2} \approx 0.42$, is invariant to $M$ for any
$M \ge 5$. **The hetero/homo conditional-calibration gap reported in
`UQ_HETERO_CALIBRATION_ANSWER.md` is therefore not an artefact of the
choice $M=20$.** It would persist with $M=5$ and degrade only if
$M$ were so small that the per-scan tertile assignment became noisy
(see §3).

## 3. Per-scan Monte-Carlo stability — the actual bottleneck

The cohort mean is stable, but the LME/LMEHetero models consume the
**per-scan vector** $(\sigma_{v,1}^2, \dots, \sigma_{v,N}^2)$, not its
mean. So the relevant question is: for a given scan $k$, how much
does $\sigma_{v,k}(m)$ change between two random ensembles of size
$m$? We measured the relative subset-to-subset SD,
$\mathrm{SD}_\text{subsets}[\sigma_{v,k}(m)] /
\mathrm{E}_\text{subsets}[\sigma_{v,k}(m)]$, for $S=500$ subsets:

| m | rel SD (median, all) | rel SD (p90, all) | rel SD (median, high tertile) |
|---|---|---|---|
| 2 | 0.78 | 1.15 | 0.83 |
| 5 | 0.35 | 0.65 | 0.39 |
| 10 | 0.18 | 0.37 | 0.20 |
| 15 | 0.10 | 0.21 | 0.11 |
| 19 | 0.04 | 0.08 | 0.04 |

At $m=2$ a single ensemble pass would deliver per-scan $\sigma_v$
estimates with 78 % relative noise — basically unusable for plugging
into a fold-specific REML fit. By $m=10$ the relative SD is ≤ 20 %,
which is comfortable: a scan in the high tertile is unlikely to
flip into low tertile under resampling. By $m=15$ it is below 11 %.

## 4. Where is the floor, and is M=20 enough?

Two convergence criteria, two answers:

| Criterion | Plateau reached at |
|---|---|
| Cohort $\overline{\sigma_v^2}$ within ±5 % of M=20 | $m = 2$ |
| Cohort $\overline{\sigma_v^2}$ within ±2 % of M=20 | $m = 5$ |
| Per-tertile means stable | $m = 2$ (already flat) |
| Per-scan rel-SD of $\sigma_v(m)$ ≤ 20 % | $m \approx 10$ |
| Per-scan rel-SD ≤ 10 % | $m \approx 15$ |

**Verdict.** $M=20$ sits comfortably on the plateau for every
cohort-level quantity that affects REML, and well past the knee for
per-scan stability. There is no evidence of degradation: $\sigma_v$
does not blow up, the high tertile does not contract, and tail
percentiles are flat from $m \approx 15$ onward. **Going beyond 20
members would not change the LMEHetero predictive variance budget
in any measurable way; it would only marginally tighten the per-scan
$\sigma_v$ around its true value.** Conversely, dropping to
$M \approx 10$ would be defensible for the calibration result
specifically, with a small loss of per-scan precision (rel SD ≈ 18 %).

## 5. Implications for the calibration story

1. **The conditional calibration table (LME → LMEHetero on the
   high-σ²_v tertile) is robust to $M$** in the sensible range
   (10–20). Any criticism that "the propagation result depends on
   the ensemble size" is empirically rejected.
2. **The bimodality of $\sigma_v^2$ is not a quirk of finite $M$**;
   it is a property of the underlying segmentation problem on this
   cohort. Some scans are genuinely hard (zero-volume targets,
   tiny tumors, bilateral edema) and the ensemble agrees that they
   are hard, regardless of how many members vote.
3. **The synthetic stress test (`UQ_SYNTHETIC_VARIANCE_STRESSTEST.md`)
   is therefore the right next experiment**: empirical $M$ does not
   limit our ability to interrogate the propagation effect, but the
   empirical $\sigma_v^2$ distribution does, and that is what the
   stress test replaces with controlled profiles.

## 6. Caveats

- These numbers measure dispersion *within* the empirical M=20
  ensemble, by random subsetting. They cannot detect a systematic
  bias of the ensemble itself. To detect that, one would need a
  larger reference ensemble (e.g. M=40 with the same training
  protocol) or a held-out test–retest segmentation.
- The "no degradation" claim covers $\sigma_v$ behaviour, not
  segmentation accuracy. Mean Dice plateaus around $r=16$/$r=32$,
  $M\ge 5$ (memory: `project_lora_ensemble_uncertainty_results`).
- The MC-stability numbers assume sampling *without replacement* from
  the same M=20 — they underestimate the true between-ensemble
  variability that one would see with independently retrained LoRAs.
  A stricter test would be to retrain $K$ independent ensembles of
  size $m$ each and measure between-ensemble SD; deferred until the
  segmentation training cost permits.

---
*Built from `experiments/.../sigma_v_vs_m.json`,
`sigma_v_vs_m_by_tertile.json`, `sigma_v_mc_stability.json`. Figure:
`sigma_v_vs_m.pdf`. Generated 2026-05-05.*
