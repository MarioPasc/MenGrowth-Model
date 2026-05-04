# Thesis ↔ Code Gap Analysis — LME / UQ Growth Prediction

**Date:** 2026-05-03
**Scope:** LME + LMEHetero models for tumor growth prediction with
uncertainty propagation from a M=20 LoRA segmentation ensemble.
**Thesis sources:**
`bachelor_thesis/.../theoretical_background/{tb_06_epistemic_uq,tb_08_gp_lme}.tex`,
`bachelor_thesis/.../methods/{growth_modeling,uncertainty_propagation}.tex`.
**Code sources:** `src/growth/models/growth/{lme_model,lme_hetero}.py`,
`src/growth/shared/{lopo,metrics,bootstrap}.py`,
`experiments/stage1_volumetric/`.

This document records (1) thesis specifications **already** implemented,
(2) gaps **fixed in this pass**, (3) gaps **still requiring action or
user decision**, and (4) implemented features **not present in the
thesis**, which the manuscript should either justify or remove.

---

## 1. Already aligned with thesis

| Spec (eq label / section) | Code location |
|---|---|
| Response transform `y = log(V+1)` (`eq:lme`) | H5 loader, `volume.transform: log1p` |
| Fixed effects `[1, t, age, sex]` (`eq:lme-covariates-methods`) | `_fit_dimension`, formula builder |
| Random intercept + slope, unstructured Ω 2×2 (`eq:lme-random`) | `re_formula="~time"`; LMEHetero `build_omega(τ₀², τ₁², ρ)` |
| Homoscedastic R = σ²·I | `LMEDimensionFit.sigma_sq` |
| Heteroscedastic `R_i = σ²_n·I + diag(σ²_v,k)` (`eq:heteroscedastic-Ky`) | `lme_hetero._build_patient_data`, `_neg_reml`, `build_Vi` |
| BLUP `û = Ω Z^T V^{-1} (y − Xβ̂)` (`eq:blup`, `eq:blup-heteroscedastic`) | `_predict_dimension`, `LMEHetero.predict` |
| REML estimation, σ²_v fixed/known | `lme_hetero._neg_reml` (REML correction at l. 301) |
| LOPO-CV evaluation | `src/growth/shared/lopo.py`, `LOPOEvaluator` |
| Coverage at 95% nominal level | `compute_coverage_at_levels` |
| Bootstrap CIs for paired comparisons | `bootstrap_metric` |
| Singular V → fallback to population mean (new patients) | `LMEHetero.predict` and `_predict_dimension` |
| Multi-start optimization | `n_restarts=5` (LMEHetero) |

---

## 2. Gaps fixed in this commit

### 2.1 Predictive variance was missing the fixed-effect uncertainty term

**Severity:** Critical for a calibration paper.

**Thesis (`uncertainty_propagation.tex`, subsec:methods-end-to-end-uq):**
the predictive variance is the sum of three components:

1. fixed-effect prediction variance,
2. random-effect posterior variance (BLUP shrinkage),
3. residual variance at t* (with `+ σ²_v*` in the heteroscedastic case).

**Code before patch:**
- `LMEGrowthModel._predict_dimension`: returned `σ² + z*ᵀ(Ω − Ω Zᵀ V⁻¹ Z Ω) z*`
  — only components (2) and (3).
- `LMEHeteroGrowthModel.predict`: same omission.

**Consequence:** The 95% prediction interval is systematically too narrow,
which biases empirical coverage **downward** and overstates apparent
sharpness. For a study whose thesis is "propagated UQ improves
calibration", this is precisely the kind of bug that can flip the
direction of the comparison or attenuate it.

**Fix applied:**
- `LMEDimensionFit` gains a `cov_beta: np.ndarray | None` field that
  stores the GLS covariance of the fixed-effect estimates
  `[β₀, β₁, β_cov₁, …]`.
- In `LMEGrowthModel._fit_dimension` (and the intercept-only fallback)
  this is extracted from the statsmodels `result.cov_params()` block via
  the new `_extract_fe_cov` helper.
- In `LMEHeteroGrowthModel.fit`, after the optimum is selected, the
  GLS information matrix `Σᵢ Xᵢᵀ Vᵢ⁻¹ Xᵢ` is recomputed at the best
  hyperparameters and inverted to give `self._cov_beta`.
- Both `predict` paths now compute `var_fe = x*ᵀ Cov(β̂) x*` with
  `x* = [1, t*, cov₁, …, cov_K]` and add it to the predictive variance.

**Note on theoretical refinement.** The Kackar–Harville (1984) /
Prasad–Rao (1990) EBLUP MSPE has an additional cross-term
`(x* − Xᵀ V⁻¹ Z Ω z*)ᵀ M⁻¹ (x* − Xᵀ V⁻¹ Z Ω z*)` rather than a simple
sum of the three blocks. The thesis explicitly states only the
three-additive-term form, so the patch matches the thesis literally.
The simple sum slightly overestimates variance relative to the cross-
term version (cross-cov is typically negative), which is the
conservative direction for coverage. If the manuscript later adopts the
EBLUP MSPE form, the change is localized to `_predict_dimension`.

**Tests:** `pytest tests/growth/test_lme_model.py
tests/growth/test_lme_hetero.py tests/growth/test_uq_propagation.py
tests/growth/test_stage1_pipeline.py tests/growth/test_growth_covariates.py`
→ 89 passed (no regressions). New behaviour: predictive variance is
strictly larger than before by `var_fe ≥ 0`, so previously-passing
positivity / monotonicity assertions remain satisfied.

---

## 3. Gaps still requiring action or decision

### 3.1 Time parameterization: ordinal index vs continuous days

**Severity:** Potentially high.

**Thesis** (`tb_08_gp_lme.tex`, `growth_modeling.tex`): `t_{ij}` is
**continuous days** (or years). The "random-slope variance scales with
t²" property in `subsec:methods-end-to-end-uq`, the log-linear
validity condition `bT ≪ 1`, and the RANO horizons (1, 2, 5 years)
all assume a continuous time axis.

**Code:** `experiments/stage1_volumetric/configs/config_uq.yaml`
defaults to `time.variable: ordinal` (timepoint index 0, 1, 2, …).
A `days_from_baseline` option exists in the loader but is not active
in the run that produced the current results
(`/media/mpascual/Sandisk2TB/.../uncertainty_propagation_volume_prediction/`).

**Why this matters for calibration:**
- With ordinal time, the heteroscedastic σ²_v values still enter R but
  the **scaling of the random-slope contribution with horizon** is
  squashed to integer steps. A 5-year extrapolation does not produce
  the larger `var_re ∝ t²` growth that the thesis claims is the main
  driver of widening intervals at long horizons.
- Per-patient sampling cadence varies (~3.6 obs/patient over 3–5 yr,
  irregular). Treating these as 0, 1, 2, … rather than as actual days
  discards uneven spacing — patients with crowded early scans look
  identical to patients with widely spaced scans.
- The Stage 2 memory note (`R²=−3.54, ordinal time blocker`) is
  consistent with this issue.

**Decision needed:** Re-run the Stage 1 UQ comparison with
`time.variable: days_from_baseline` (or `years_from_baseline` to keep
β₁ and τ₁² on a clinically interpretable scale). Compare the
calibration table (cov@95, sharpness, CRPS) before vs after. **Until
this is done, the empirical coverage numbers in the current results
folder cannot be reported as a faithful test of the thesis.**

### 3.2 Bootstrap resamples 2,000 vs 10,000

**Severity:** Low.

Thesis (`subsec:validation`) specifies B = 10,000 bootstrap resamples.
`config_uq.yaml` uses `bootstrap.n_samples: 2000`. This is a
1-line config change with negligible cost; recommend bumping to 10,000
before the final paper run for consistency with the methods text.

### 3.3 Statistical test: Wilcoxon signed-rank vs paired permutation

**Severity:** Low–medium.

Thesis specifies **paired Wilcoxon signed-rank** plus Cohen's d.
Code uses `paired_permutation_test` (10,000 permutations) — also valid
and arguably more powerful, but a different test.

**Recommendation:** Add a thin `paired_wilcoxon_test` helper alongside
the permutation test in `src/growth/shared/bootstrap.py` and report
both in the comparison table; pick one as primary in the manuscript
and footnote the other for transparency. Cohen's d on per-patient
errors is also not currently computed and should be added.

### 3.4 LOPO target convention: final timepoint vs all held-out timepoints

**Severity:** Medium — potentially affects what coverage is being
measured.

Thesis (`subsec:validation`) states: "each patient's **final
timepoint** is the held-out target." The current `LOPOEvaluator`
appears to evaluate against all held-out patient timepoints (the
held-out patient is fully out of training, but its full trajectory is
predicted from `n_condition` early observations). These are different
estimands:

- Thesis estimand: P(coverage at the **single** clinically relevant
  endpoint per patient), with conditioning on all earlier scans.
- Code estimand: a mixture over multiple horizons within each
  held-out patient.

**Action required:** verify in `lopo_evaluator.py` and the
stage1 runner whether the metric is restricted to the final timepoint
or pooled across all held-out timepoints, and align with thesis. If
both are useful, report the final-timepoint table as primary
(thesis-aligned) and the pooled table as secondary.

### 3.5 RANO clinical decision metric

**Severity:** Low for the methodology comparison, but the thesis
explicitly defines `P(V(t*) > 1.4 · V_baseline)` evaluated at horizons
1, 2, 5 years (`eq:exceedance`). Not implemented in the current code.
Add a small post-hoc evaluator that, for each LOPO fold, computes the
exceedance probability at fixed horizons from the predictive Gaussian.
Report calibration of these probabilities against observed RANO-
progression labels.

### 3.6 Log-linear validity check

**Severity:** Low (theoretical sanity check).

Thesis (`eq:log-linear-approx`) requires `(bT)²/2 < σ²_v floor`.
Add a one-shot diagnostic that, given fitted `β̂₁` and the σ²_v
distribution, prints whether the inequality holds for T = 5 yr.
Goes in `experiments/stage1_volumetric/stats/` as a verification
artifact, not in the modelling code.

---

## 4. Code features not present in the thesis

These are implemented and active in the code but the manuscript does
not discuss them. For each, decide whether to (a) add to the
manuscript with justification, or (b) drop from the run to keep the
methodology focused.

| Feature | Code location | Recommendation |
|---|---|---|
| `ScalarGPHetero`, `HGPHeteroModel`, `HGP_Gompertz_Hetero` (8-model ablation rather than thesis's 2-model homo vs hetero LME) | `models/growth/{scalar_gp_hetero,hgp_hetero}.py`; `engine/model_registry.py` | Likely worth keeping as an ablation table (more complete picture). Add a paragraph to methods explaining the GP variants and citing the relevant kernels. |
| CRPS, NLL, multi-level coverage curves | `shared/metrics.py`, `shared/calibration_plots.py` | Add to the thesis. CRPS is a strictly proper scoring rule and strengthens the calibration argument. Multi-level coverage is a more informative figure than a single 95% number. |
| PIT histogram plot | `shared/calibration_plots.py` | Add to thesis. Augment with a KS test of PIT vs Uniform(0,1) to make the visual quantitative. |
| Variance decomposition module | `evaluation/variance_decomposition.py` | Used by Stage system, not by Stage 1 UQ comparison. Either remove from `config_uq.yaml` or document its role. |
| Floor variance `1e-6` on σ²_v | `LMEHeteroGrowthModel.floor_variance` | Add a single sentence to methods explaining the numerical floor. |
| GLS covariance via REML information vs sandwich | (current patch uses `(Σ Xᵀ V⁻¹ X)⁻¹`) | Consistent with thesis derivation. No change needed. |
| `median_mad`, `mask_mean` ensemble estimators (alternative to `mean_std`) | `data/uncertainty_loader.py` | The thesis only mentions mean/variance over the ensemble. Either drop these from the active config or add a short ablation. |

---

## 5. Recommended next actions, in order of impact

1. **[Done]** Add fixed-effect uncertainty term to LME and LMEHetero
   predictive variance.
2. **Re-run** Stage 1 UQ with `time.variable: days_from_baseline`
   (or `years_from_baseline`); regenerate the comparison table.
3. **Verify and document** the LOPO target convention; align with the
   thesis "final-timepoint" estimand or add it as the primary table.
4. Bump `bootstrap.n_samples` to 10,000 in `config_uq.yaml`.
5. Add `paired_wilcoxon_test` and Cohen's d helpers; report
   alongside permutation test.
6. Add RANO exceedance-probability calibration metric.
7. Update the manuscript methods to acknowledge CRPS / NLL / PIT /
   multi-level coverage as part of the calibration battery, or drop
   them from the active reporting.
