# Smooth σ²_v Shape-Shift Stress Test

**Status:** design document. No code yet.
**Date:** 2026-05-07.
**Companion docs:**
`UQ_HETERO_CALIBRATION_ANSWER.md`,
`UQ_CALIBRATION_STORY.md`,
`UQ_SYNTHETIC_VARIANCE_STRESSTEST.md` (Profiles A–E),
`UQ_SYNTHETIC_VARIANCE_STRESSTEST_RESULTS.md`,
`UQ_THESIS_GAP_ANALYSIS.md`.

---

## 0. Why this experiment

The Profile A–E sweep (`UQ_SYNTHETIC_VARIANCE_STRESSTEST_RESULTS.md`)
parameterises σ²_v dispersion **discretely**: constant (A), bimodal at
fixed mixture weights (B, C), log-normal at five τ values (D),
empirical (E). It established two facts:

1. The headline LME→LMEHetero high-tertile gain decomposes into ~91 %
   structural + ~9 % genuine σ²_v propagation.
2. Informativeness of σ²_v matters: random dispersion (B, C, D)
   widens the wrong intervals; only empirical σ²_v (E) lowers
   high-tertile IS@95.

What it did **not** do:

- Trace the calibration metric landscape *continuously* as the σ²_v
  distribution morphs from "all mass at zero" (clean ensemble,
  hetero ≡ homo) through "all mass uniform" (random dispersion,
  worst-case for the propagation claim) to "all mass at high
  variance" (every scan flagged uncertain, hetero must widen
  everything).
- Plot the **REML budget identity** $\hat\sigma^2_{n}(\alpha) +
  \overline{\sigma^2_v}(\alpha) \approx \mathrm{const}$ as a smooth
  curve.
- Quantify the **per-tertile sharpness fan-out** as the σ²_v shape
  shifts.

This document specifies a single-knob, smooth-family stress test
that fills those gaps. It is the cleanest visualisation of the
"hetero re-allocates uncertainty where σ²_v rises" claim and is
intended to become the headline figure for the Stage 1 UQ chapter.

---

## 1. The σ²_v shape family

We parameterise the σ²_v distribution by a single shape parameter
$\alpha \in [-1, +1]$ that interpolates smoothly across three
limits:

| α | Distribution shape | Interpretation |
|---|---|---|
| **−1** | mass concentrated at 0 (Beta(1, b_max)) | "perfect ensemble" — every scan well-segmented |
| **0** | uniform on $[0, \sigma^2_\max]$ (Beta(1, 1)) | maximally non-informative dispersion |
| **+1** | mass concentrated at $\sigma^2_\max$ (Beta(a_max, 1)) | "everything is hard" — every scan flagged uncertain |

### 1.1 Family choice — Beta on $[0, \sigma^2_\max]$

For each α, sample $\sigma^2_{v,k} = \sigma^2_\max \cdot u_k$ where
$u_k \sim \mathrm{Beta}(a(\alpha), b(\alpha))$ and

$$
a(\alpha) = \begin{cases} 1 & \alpha \le 0 \\ 1 + s\alpha & \alpha > 0 \end{cases}, \qquad
b(\alpha) = \begin{cases} 1 - s\alpha & \alpha \le 0 \\ 1 & \alpha > 0 \end{cases},
$$

with a steepness constant $s$ chosen so that at $|\alpha| = 1$ the
distribution is sharply peaked (e.g. $s = 9$ → Beta(1, 10) at
α = −1, Beta(10, 1) at α = +1, Beta(1, 1) = uniform at α = 0).
This produces a *continuous one-parameter family* with monotonic
shape change.

**Why Beta and not log-normal+τ.** Profile D used log-normal with
$\tau$ as the dispersion knob, which probes "how much spread", not
"where is the peak". Beta on a bounded support directly probes the
peak location, which is the question this stress test wants to ask.

### 1.2 Disentangling shape from magnitude

The mean of $\sigma^2_v$ under this family changes with α
($\mathbb{E}[\mathrm{Beta}(a, b)] = a/(a+b)$):
α = −1 → 0.09, α = 0 → 0.5, α = +1 → 0.91 (in units of
$\sigma^2_\max$). Two analyses:

- **Free-mean sweep** (primary): keep $\sigma^2_\max$ fixed, let
  $\overline{\sigma^2_v}$ vary with α. Probes the *joint* shape +
  magnitude effect on REML's absorption budget.
- **Fixed-mean sweep** (control): rescale each draw so that
  $\overline{\sigma^2_v}(\alpha) = c$ is constant across α. Isolates
  the *peak-location* effect from magnitude.

Run both. The difference shows how much of the calibration shift is
"REML absorbed more mean noise" vs "the high-σ²_v patients are
different ones at different α".

### 1.3 Informativeness — random vs informative assignment

Two assignment protocols per α:

- **Random assignment** (negative control): permute the sampled
  $\{\sigma^2_{v,k}\}$ randomly across scans. This isolates the
  shape effect from σ²_v's correlation with residual difficulty.
- **Rank-preserving assignment** (positive condition): rank scans
  by their *empirical* σ²_v and assign the synthetic σ²_v values in
  the same rank order. This preserves the *ordering* of which scan
  is hardest, rescaling only the magnitudes per α. This is the
  "informative shape sweep" — the analogue of Profile E for a
  continuous family.

The rank-preserving sweep is the **scientifically interesting**
condition: it asks "if the segmentation pipeline were ever-better
(α → −1) or ever-worse (α → +1) but kept ranking patients by
difficulty correctly, how would calibration shift?".

### 1.4 Function specification (no code)

```text
def sample_sigma_v_shape(
    alpha: float,                   # shape knob in [-1, 1]
    n: int,                         # cohort size (= 173 scans)
    sigma_v_sq_max: float,          # support upper bound
    steepness: float = 9.0,         # peak sharpness at |α|=1
    fixed_mean: float | None = None,# rescale to this mean if given
    assignment: str = "rank",       # 'random' | 'rank' | 'identity'
    rank_reference: ndarray | None, # required if assignment='rank'
    seed: int,
) -> ndarray:                       # shape (n,) — σ²_v per scan
    """
    Draw n samples from Beta(a(α), b(α)) on [0, σ²_v_max], optionally
    rescale to fixed mean, and assign to scans either randomly or
    in the rank order of an external reference (empirical σ²_v).
    """
```

The function lives in
`experiments/stage1_volumetric/synthetic_uq/sample_profiles.py`
alongside the existing Profile A–E samplers.

---

## 2. Sweep grid

| Axis | Values | n |
|---|---|---|
| α (shape) | {−1.0, −0.75, −0.5, −0.25, 0, 0.25, 0.5, 0.75, 1.0} | 9 |
| Mean condition | {free, fixed=0.5·σ²_max} | 2 |
| Assignment | {random, rank-preserving} | 2 |
| Seed | 1..R, R = 50 | 50 |
| Model | {LME (statsmodels), LMEHetero@σ²_v=0, LMEHetero@injected} | 3 |

Total LOPO LMEHetero fits: 9 × 2 × 2 × 50 × 2 = 3 600.
LME (statsmodels) is cached once. **LMEHetero@σ²_v=0 should be
cached once across α/mean/assignment** (it does not depend on the
injected σ²_v) — this gives the controlled-homo baseline from §2 of
the parent answer. Per-fold runtime ≈ 5 s × 56 folds ≈ 5 min per
LMEHetero LOPO run; full sweep ≈ 300 CPU-hours, embarrassingly
parallel.

`sigma_v_sq_max` is set to the empirical 95th percentile of σ²_v
(≈ 1.5 in log-volume units²). This keeps the support physically
plausible without including the segmentation-failure outliers.

Tertile boundaries are anchored to **empirical** σ²_v as in
Profile A–E so the patient strata are stable across α and
cross-condition comparisons are paired by patient.

---

## 3. Headline visualisations

The point of this sweep is *visual*: a continuous shape knob on the
x-axis, calibration metrics on the y-axis, with overlays that show
the mechanism. Six figures, in order of importance.

### 3.1 Fig 1 — σ²_v density curves stacked along α

**Purpose:** show what the shape knob actually does.

**Layout:** ridgeline plot. Y-axis = α (9 levels). X-axis = σ²_v on
$[0, \sigma^2_\max]$. Each row is a 1D KDE of the sampled σ²_v at
that α (from one fixed seed). Annotate the empirical distribution
as a dashed line for reference.

**Reads:** the peak slides smoothly from 0 (top, α = −1) through
uniform (middle, α = 0) to $\sigma^2_\max$ (bottom, α = +1).

### 3.2 Fig 2 — Three-metric landscape vs α

**Purpose:** the headline result. How do R², IS@95, cov@95 shift as
the σ²_v shape morphs?

**Layout:** 3 stacked panels (one per metric). X-axis = α. Each
panel has up to four curves:

- LMEHetero@injected, free-mean, **rank-preserving** assignment
  (the informative condition);
- LMEHetero@injected, free-mean, **random** assignment (negative
  control);
- LMEHetero@σ²_v=0 (controlled homo baseline, horizontal line);
- LME statsmodels (literature-default homo, horizontal line).

Shaded bands: 95 % bootstrap CI across the 50 seeds.

**Reads:**

- R²_log should be ≈ flat across α for hetero — propagation does not
  change point predictions.
- IS@95 should be **U-shaped** for the rank-preserving condition with
  minimum near the empirical α (where σ²_v carries the most
  information). Random assignment should be flat or worse.
- cov@95 (marginal) should drift mildly with α — the manuscript
  should NOT rely on this; the structural metric is per-tertile
  cov@95 (Fig 3).

### 3.3 Fig 3 — Per-tertile cov@95 heatmap

**Purpose:** direct visualisation of the conditional thesis claim.

**Layout:** 3 × 9 heatmap. Rows = σ²_v_target tertile (low, mid,
high — anchored to empirical). Columns = α. Cells coloured by
cov@95. One heatmap per condition (random, rank-preserving) and per
homo baseline (statsmodels, LMEHetero@σ²_v=0) for direct contrast.

**Reads:** rank-preserving condition should show a *diagonal-ish*
pattern: at α = −1 hetero matches homo at all tertiles; at α = +1
hetero widens *everywhere* and over-covers; near α = 0
(uniform/random) hetero rescues the high-tertile coverage.

### 3.4 Fig 4 — Per-tertile CI width fan plot

**Purpose:** redistribution mechanism made visible.

**Layout:** X-axis = α. Y-axis = CI width. Three lines per condition,
one per tertile (low, mid, high), for LMEHetero. Two horizontal
reference lines for LME and LMEHetero@σ²_v=0.

**Reads:** the three tertile lines fan out as α grows — narrow on
low, wide on high. The fan opening is the "moves sharpness where
the data justifies it" effect, made continuous.

### 3.5 Fig 5 — REML budget identity curve

**Purpose:** verify that REML's absorption mechanism is doing what
§5.1 of `UQ_CALIBRATION_STORY.md` predicts.

**Layout:** X-axis = α. Two lines:

- $\hat\sigma^2_{n}(\alpha)$ from LMEHetero REML (mean across folds
  and seeds);
- $\overline{\sigma^2_v}(\alpha)$ from the synthetic samples;
- Their sum (should be ≈ flat at $\sigma^2_{n,\text{homo}} \approx
  0.95$).

**Reads:** the two should trade off cleanly along α — REML absorbs
the cohort-mean σ²_v into a smaller σ²_n. If they don't sum to a
flat line, the "REML absorbs the mean" story is incomplete and we
need to look at the random-effect variance budget too.

### 3.6 Fig 6 — Sharpness vs σ²_v_target scatter, faceted by α

**Purpose:** the per-scan picture.

**Layout:** 3 × 3 facets (one per α ∈ {−1, 0, +1} × condition).
X-axis = $\sigma^2_{v,*}$ (target's empirical σ²_v, log scale).
Y-axis = predictive CI width at that target. Each point = one of the
56 LOPO folds. Overlay the homo baseline as a horizontal line.

**Reads:** at α = −1, hetero scatter sits below homo (sharpens
everywhere). At α = 0, scatter has no slope (random dispersion). At
α = +1 with rank-preserving assignment, scatter has a positive
slope and crosses homo at $\sigma^2_{v,*} \approx \overline{\sigma^2_v}$.
This panel is the discrete version of Fig 4 made per-fold.

### 3.7 Fig 7 (optional) — Pareto frontier IS@95 vs cov@95

**Purpose:** judging the "good calibration" Pareto front.

**Layout:** scatter of (cov@95, IS@95) per (α, condition, model)
cell. The bottom-right corner (high coverage, low IS) is the goal.

**Reads:** rank-preserving hetero traces a curve that hugs the
bottom-right; random-assignment hetero spirals outward (wider
intervals at the same coverage = wasted variance budget).

---

## 4. Falsifiable predictions (per visualisation)

| # | Prediction | Figure | What falsifies it |
|---|---|---|---|
| 1 | R² flat in α for both rank-preserving and random | Fig 2 panel 1 | R² varies systematically with α → propagation affects mean, not just variance — implementation bug. |
| 2 | IS@95 (rank-preserving) U-shaped with min near empirical α | Fig 2 panel 2 | Monotonic in α → informativeness does not localise; effect is just dispersion. Re-examine assignment protocol. |
| 3 | Per-tertile cov@95 diagonal pattern in rank-preserving condition | Fig 3 | Flat across α on each tertile → hetero is not redistributing; could indicate σ²_v is too small to matter or REML is converging to a different local optimum. |
| 4 | CI width fan opens monotonically with α (rank-preserving) | Fig 4 | Fan does not open or opens for random too → §5.1 of UQ_CALIBRATION_STORY.md is incomplete. |
| 5 | $\hat\sigma^2_n + \overline{\sigma^2_v}$ ≈ flat in α | Fig 5 | Sum drifts with α → REML is doing something more complicated than mean absorption (e.g. moving variance into Ω). |
| 6 | LMEHetero@σ²_v=0 ≡ LME (statsmodels) on all metrics | Fig 2 horizontal lines | Visible gap → implementation drift between MixedLM and the custom REML. Quantifies the "structural baseline" of Profile A. |
| 7 | Random-assignment hetero strictly dominated by rank-preserving on IS@95 | Fig 7 | Random matches rank → informativeness does not matter; falsifies Profile B/E contrast. |

---

## 5. What this experiment delivers for the manuscript

1. **A single, continuous, single-knob figure** (Fig 2) that
   replaces the discrete Profile A–E table as the headline
   propagation visual.
2. **Fig 3** as the structural "conditional calibration shifts as
   σ²_v shifts" plot — directly answers "does the hetero model
   move the variance where measurement uncertainty rises?".
3. **Fig 5** as the mechanistic explanation: REML absorbs the mean,
   the trade-off is visible in real time as α slides.
4. **Quantitative bound on the structural baseline** via the
   LMEHetero@σ²_v=0 horizontal line in Fig 2 — closes the
   conflation issue from §2 of the parent answer.

Combined, these supersede the Profile A–E table (which becomes a
checkpoint result, kept for reproducibility) and produce a
substantially stronger Stage 1 UQ chapter.

---

## 6. Implementation checklist (deferred)

- [ ] Implement `sample_sigma_v_shape(α, ...)` in
  `experiments/stage1_volumetric/synthetic_uq/sample_profiles.py`.
- [ ] Implement `LMEHetero@σ²_v=0` baseline runner (just calls the
  existing `LMEHeteroGrowthModel` with σ²_v ≡ floor_variance).
- [ ] Add per-tertile R²_log to `conditional_calibration.py` (gap
  from §1 of the parent answer).
- [ ] Add ΔIS@95 + per-tertile bootstrap p-value to the comparison
  pipeline.
- [ ] Run the sweep (≈ 300 CPU-hours, Picasso CPU partition).
- [ ] Generate Figures 1–7 from the aggregated outputs.
- [ ] Write up as a manuscript appendix; promote Fig 2 + Fig 3 to
  the main text.

---

## 7. Caveats

- **Tertile boundaries are anchored to empirical σ²_v across all α.**
  This is the right choice for paired comparisons but conflates
  "patients with high empirical σ²_v" with "hard patients" at α ≠ 0.
  An alternative stratification by injected σ²_v would isolate the
  shape effect at the cost of comparability.
- **Ordinal time still in use.** The propagation effect is expected
  to grow under continuous time (var_re ∝ t²). This sweep should be
  re-run under `time.variable: years_from_baseline` once dates are
  wired (`UQ_THESIS_GAP_ANALYSIS.md` §3.1).
- **σ²_max = empirical p95 ≈ 1.5** excludes the segmentation-failure
  tail. A separate ablation should sweep σ²_max to probe how the
  conclusions depend on this support choice.
- **Computational cost (≈ 300 CPU-hours).** Reduce by pruning the
  α grid to {−1, −0.5, 0, 0.5, +1} and seeds to 25 if the wall-clock
  budget is tight; results should be qualitatively unchanged.

---

*End of design document. No code changes yet.*
