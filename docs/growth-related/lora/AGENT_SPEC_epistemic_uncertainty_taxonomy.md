# Agent Specification: Epistemic Uncertainty Taxonomy Integration

**Author:** Mario Pascual González (via Claude)
**Date:** 2026-04-09
**Priority:** High — affects both thesis narrative and downstream LME integration
**Scope:** Documentation updates + new analysis code on existing LoRA ensemble results
**Prerequisite data:** Completed ensemble inference CSVs at `r∈{4,8,16}`, M=20, seed=42

---

## 0. Motivation and theoretical background

A recent position paper — Jiménez, Jürgens & Waegeman (2026), "Epistemic uncertainty estimation methods are fundamentally incomplete" (arXiv:2505.23506v3) — provides a formal argument that second-order epistemic uncertainty estimators (Deep Ensembles, MC Dropout, Bayesian NNs, etc.) are incomplete in two specific, mathematically characterizable ways:

### 0.1 Bias contamination of aleatoric estimates

The classical variance-attenuation loss yields a biased aleatoric estimator. For a heteroscedastic model outputting mean $\hat{f}(x)$ and variance $\hat{\sigma}^2(x)$:

$$\mathbb{E}[\hat{\sigma}^2(x)] = \bigl(f(x) - \hat{f}(x)\bigr)^2 + \sigma^2(x) \tag{B.1, Jiménez et al.}$$

where $f(x)$ is the true conditional mean and $\sigma^2(x)$ is the true aleatoric noise. The squared bias term $(f(x) - \hat{f}(x))^2$ inflates what the model reports as aleatoric uncertainty. Conversely, because the bias is absorbed into the aleatoric term, the epistemic estimate is deflated — the model appears confident precisely where it is wrong.

**Relevance to our pipeline:** Our LoRA ensemble does not output a heteroscedastic variance head per se — volumes are computed from hard masks. However, the *same mechanism* operates at the volume level: if all M ensemble members are systematically biased (e.g., all over-segment due to the shared frozen encoder's inductive bias, or all under-segment for a particular scanner contrast), the inter-member volume variance $\text{Var}_\gamma[V]$ remains small (members agree on the wrong answer) while the true error $|V_{\text{pred}} - V_{\text{true}}|$ is large. The LME then receives a tight $\sigma_{v,k}$ for a systematically wrong volume measurement.

### 0.2 Procedural-only variance capture

The paper decomposes estimation variance via the law of total variance:

$$\text{Var}_{\mathcal{D}_N, \gamma}(\hat{y}|x) = \underbrace{\mathbb{E}_{\mathcal{D}_N}\!\bigl[\text{Var}_\gamma(\hat{y}|x) \,|\, \mathcal{D}_N\bigr]}_{\text{procedural uncertainty}} + \underbrace{\text{Var}_{\mathcal{D}_N}\!\bigl(\mathbb{E}_\gamma[\hat{y}|x] \,|\, \mathcal{D}_N\bigr)}_{\text{data uncertainty}} \tag{Eqn. 8}$$

A Deep Ensemble — or any method that trains multiple models with different random seeds on the **same** dataset — captures only the first term. The data uncertainty component, which measures how predictions would change under different training datasets from the same distribution, is entirely absent.

**Relevance:** Our LoRA ensemble is structurally identical to a Deep Ensemble: M=20 LoRA adapters, all trained on the same BraTS-MEN 2024 dataset, differing only in random seed $\gamma$ (weight initialization + augmentation ordering). Therefore, the ensemble volume standard deviation $\sigma_{v,k} = \text{Std}_\gamma[V^{(m)}_k]$ captures **only procedural uncertainty**. With a training set of ~800 BraTS-MEN subjects and only ~112 MenGrowth temporal observations downstream, the data uncertainty component is potentially substantial.

### 0.3 Five-source taxonomy

The paper identifies five sources of epistemic uncertainty (their Figure 2):

| Source | Definition | Our pipeline analogue |
|--------|-----------|----------------------|
| **Approximation error** | $\mathcal{L}(\hat{p}^*) - \mathcal{L}(p)$: gap because $p(y|x) \notin \mathcal{H}$ | BrainSegFounder's hypothesis space limitations for meningioma boundary delineation |
| **Estimation bias** | $\mathcal{L}(\bar{p}^*) - \mathcal{L}(\hat{p}^*)$: systematic deviation from Bayes-optimal due to regularization + finite data | LoRA regularization (low-rank constraint) + frozen encoder bias + Gompertz/linear mean function constraint in LME |
| **Procedural uncertainty** | $\mathbb{E}_{\mathcal{D}_N}[\text{Var}_\gamma(\hat{y}|x) | \mathcal{D}_N]$: sensitivity to random seeds | **What we currently measure** — LoRA ensemble volume variance |
| **Data uncertainty** | $\text{Var}_{\mathcal{D}_N}(\mathbb{E}_\gamma[\hat{y}|x] | \mathcal{D}_N)$: sensitivity to training data | Partially captured by LOPO-CV (each fold trains on $\mathcal{D}_{N-1}$), but **not propagated** per-fold to LME |
| **Distributional uncertainty** | Train/test distribution shift | Multi-scanner domain shift (BraTS-MEN → Andalusian cohort); ComBat-mitigated, not eliminated |

### 0.4 Key references

- Jiménez, S., Jürgens, M., & Waegeman, W. (2026). Position: Epistemic uncertainty estimation methods are fundamentally incomplete. arXiv:2505.23506v3.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS.
- Mühlematter, D.J. et al. (2024). LoRA-Ensemble: Efficient Uncertainty Modelling for Self-Attention Networks. arXiv:2405.14438.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer. (Ch. 7: bias-variance decomposition)
- Zhang, W. et al. (2024). One Step Closer to Unbiased Aleatoric Uncertainty Estimation. AAAI 2024.

---

## 1. Task A — Documentation update: UNCERTAINTY_SEGMENTATION_SUMMARY.md

### 1.1 What to change

Add a new **Section 8.1** (after the existing Section 8 "Theoretical connection to the growth pipeline") titled:

**"8.1 Epistemic uncertainty taxonomy and known limitations"**

### 1.2 Exact content to add

Insert the following section after the existing Section 8 content (after the equation for $P(V(t^*) > V_{\text{critical}})$) and before Section 9 (Key references):

```markdown
### 8.1 Epistemic uncertainty taxonomy and known limitations

The LoRA ensemble's uncertainty estimate must be interpreted within the formal taxonomy
of epistemic uncertainty sources established by Jiménez, Jürgens & Waegeman (2026,
arXiv:2505.23506v3). Their analysis, grounded in the bias-variance decomposition of
Bregman divergences (Hastie et al., 2009; Gruber et al., 2023), demonstrates that
second-order methods — including Deep Ensembles, to which the LoRA ensemble is
structurally equivalent — are fundamentally incomplete in two ways.

**Procedural-only variance.** The LoRA ensemble trains $M$ adapters on the same
BraTS-MEN 2024 dataset with different random seeds $\gamma_m$, making the ensemble volume
variance

$$\sigma^2_{v,k} = \text{Var}_\gamma\!\bigl[V^{(m)}_k\bigr] = \frac{1}{M-1}\sum_{m=1}^{M}\bigl(V^{(m)}_k - \bar{V}_k\bigr)^2$$

an estimator of the **procedural component** of epistemic uncertainty only (first term of
the law of total variance decomposition). The **data component** — how volume estimates
would vary under different training datasets drawn from the same distribution — is
entirely absent. At $N_{\text{train}} \approx 800$ BraTS-MEN subjects, data uncertainty is likely
smaller than procedural uncertainty for the segmentation task itself, but for downstream
growth modeling on $N \approx 31\text{–}58$ MenGrowth patients, this missing component is
potentially substantial.

**Bias contamination.** When the ensemble mean segmentation is systematically biased
(e.g., from the frozen encoder's inductive bias, scanner-specific artifacts, or the
low-rank LoRA constraint), all members agree on the wrong answer. The inter-member
variance remains small while the true prediction error is large. This bias is invisible
to the variance-based uncertainty estimate and is absorbed by the LME's residual
variance $\sigma^2_\varepsilon$, inflating what the growth model treats as irreducible noise.

**Practical consequence.** The uncertainty propagated to the LME/GP via $\sigma_{v,k}$
should be interpreted as a **lower bound** on the total segmentation-derived epistemic
uncertainty. We report empirical diagnostics (bias vs. ground truth, calibration
coverage) in the analysis outputs to quantify the magnitude of the missing components.

The following table maps the five epistemic uncertainty sources identified by
Jiménez et al. (2026) to specific elements of our pipeline:

| Source | Pipeline element | Status |
|--------|-----------------|--------|
| Approximation error | BrainSegFounder hypothesis space | Not quantifiable without oracle |
| Estimation bias | LoRA regularization + frozen encoder | Diagnosed via LOPO-CV bias metric |
| Procedural uncertainty | LoRA ensemble variance $\sigma_{v,k}$ | **Measured** |
| Data uncertainty | LOPO-CV fold variation | Partially captured, not propagated per-fold |
| Distributional uncertainty | BraTS-MEN → Andalusian shift | ComBat-mitigated |

**Mitigating factor — LME random effects.** The patient-level random intercept and
slope in the LME partially absorb systematic per-patient segmentation bias, because
bias correlates with scanner and scanner correlates with patient. This does not
eliminate the problem but provides a structural mitigation.

**Mitigating factor — Gompertz/linear mean function.** The strong parametric
constraint on the mean function increases estimation bias but reduces variance
(the classical bias-variance trade-off). At $N = 31\text{–}58$, this trade-off is favorable:
LME's current superiority over more flexible GP variants ($R^2_{\log} \approx 0.387$ vs.
ScalarGP $\approx 0.077$) is a direct consequence, consistent with the paper's formal
argument that regularization-induced bias reduces the variance term of Eqn. 7 while
increasing the bias term.
```

### 1.3 Additional references to add to Section 9

Append the following entries:

```markdown
8. Jiménez, S., Jürgens, M., & Waegeman, W. (2026). Position: Epistemic uncertainty
   estimation methods are fundamentally incomplete. arXiv:2505.23506v3.
9. Gruber, C. et al. (2023). Sources of uncertainty in machine learning — a
   statistician's view. arXiv preprint.
10. Zhang, W. et al. (2024). One Step Closer to Unbiased Aleatoric Uncertainty
    Estimation. AAAI 2024, 38(15), 16857–16864.
```

### 1.4 Verification

After editing, confirm:
- The new section number (8.1) does not collide with existing sections.
- The table renders correctly in markdown preview.
- All equation references ($\sigma_{v,k}$, $\sigma^2_\varepsilon$) match existing notation in the document.
- The existing Section 8 equation chain ($p(\mathbf{s}|\mathbf{x}) \to p(V|\mathbf{s}) \to p(V(t^*)|\{V_k\})$) flows naturally into the new taxonomy section.

---

## 2. Task B — New analysis script: epistemic uncertainty diagnostics

### 2.1 Purpose

Compute, from the existing ensemble volume CSVs, the empirical diagnostics recommended by Jiménez et al. (2026) — specifically:

1. **Bias metric** per scan: $\text{bias}_k = \bar{V}_k^{(\text{ensemble})} - V_k^{(\text{GT})}$ on BraTS-MEN test set (where ground truth segmentation volume is available).
2. **Bias vs. ensemble std scatter** to visually diagnose bias contamination (analogous to their Figure 8).
3. **Calibration coverage** of ensemble-derived confidence intervals on volume.
4. **Taxonomy summary table** per LoRA rank.
5. **Cross-rank comparison** of procedural uncertainty magnitude.

### 2.2 Location and file structure

```
experiments/uncertainty_segmentation/engine/epistemic_taxonomy.py
```

This follows the existing convention in `engine/` for analysis scripts (e.g., `statistical_analysis.py`, `convergence_analysis.py`). A CLI wrapper is not needed — the script is called from within `run_evaluate.py` or as a standalone analysis.

### 2.3 Input data

The script reads from existing outputs. For each rank configuration `r{rank}_M20_s42/`:

**File 1 — Evaluation volumes (BraTS-MEN test set with GT):**
```
{run_dir}/evaluation/per_member_test_dice.csv
```
Columns: `member_id`, `scan_id`, `dice_tc`, `dice_wt`, `dice_et`, `dice_mean`, `volume_pred`

**File 2 — Ensemble evaluation (BraTS-MEN test set with GT):**
```
{run_dir}/evaluation/ensemble_test_dice.csv
```
Columns include: `scan_id`, predicted and ground truth WT volumes.

**File 3 — MenGrowth volumes (no GT for volume, but has per-member volumes):**
```
{run_dir}/volumes/mengrowth_ensemble_volumes.csv
```
Columns as documented in UNCERTAINTY_SEGMENTATION_SUMMARY.md §6.3.

> **Agent note:** Inspect these CSVs to confirm exact column names before coding. The names above are from the project documentation; the actual implementation may differ slightly. Use `pd.read_csv(..., nrows=5)` to check.

### 2.4 Implementation specification

```python
"""
Epistemic uncertainty taxonomy diagnostics.

Computes bias, calibration, and variance decomposition metrics
following the framework of Jiménez, Jürgens & Waegeman (2026,
arXiv:2505.23506v3) to characterize what the LoRA ensemble
uncertainty estimate captures and what it misses.

Location: experiments/uncertainty_segmentation/engine/epistemic_taxonomy.py
"""
```

#### 2.4.1 Function: `compute_volume_bias`

```python
def compute_volume_bias(
    per_member_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    volume_col_gt: str = "volume_gt",       # verify actual column name
    volume_col_pred: str = "volume_pred",    # verify actual column name
) -> pd.DataFrame:
    """
    Compute per-scan bias and ensemble std on the BraTS-MEN test set.

    For each scan k:
        bias_k = V_bar_ensemble_k - V_gt_k
        std_k  = Std_gamma[V^(m)_k]   (procedural uncertainty)

    The relationship between |bias_k| and std_k diagnoses bias contamination:
    if |bias| >> std, the ensemble is confidently wrong (Jiménez et al., §3.1).

    Returns DataFrame with columns:
        scan_id, volume_gt, volume_ensemble_mean, volume_ensemble_std,
        bias, abs_bias, bias_to_std_ratio
    """
```

**Logic:**
1. From `per_member_df`, group by `scan_id`, compute mean and std of `volume_pred` across members.
2. Merge with `ensemble_df` on `scan_id` to get `volume_gt`.
3. Compute: `bias = volume_ensemble_mean - volume_gt`, `abs_bias = |bias|`, `bias_to_std_ratio = abs_bias / (volume_ensemble_std + 1e-8)`.
4. Return the merged DataFrame.

**Log-volume variant:** Also compute all metrics on $\log(V + 1)$ scale, since the downstream LME operates in log-volume space. Add columns `logvol_bias`, `logvol_abs_bias`, `logvol_bias_to_std_ratio`.

#### 2.4.2 Function: `compute_calibration_coverage`

```python
def compute_calibration_coverage(
    per_member_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    nominal_levels: list[float] = [0.50, 0.80, 0.90, 0.95],
) -> pd.DataFrame:
    """
    Compute empirical coverage of ensemble-derived confidence intervals.

    For each nominal level alpha:
        1. Compute ensemble mean and std per scan.
        2. Construct interval: [mean - z_{alpha/2} * std, mean + z_{alpha/2} * std]
           using a t-distribution with df=M-1 (more appropriate than Gaussian for M=20).
        3. Check whether V_gt falls within the interval.
        4. Report empirical coverage = fraction of scans covered.

    If coverage < nominal, the ensemble underestimates total uncertainty,
    consistent with the Jiménez et al. argument that procedural variance
    alone is an incomplete epistemic estimate.

    Returns DataFrame with columns:
        nominal_level, z_or_t_multiplier, empirical_coverage, n_scans,
        n_covered, coverage_deficit
    """
```

**Important:** Use `scipy.stats.t.ppf` with `df=M-1` for the critical values, not the Gaussian. At M=20, the difference is non-trivial (t_{19,0.975} = 2.093 vs z_{0.975} = 1.960).

#### 2.4.3 Function: `compute_cross_rank_procedural_variance`

```python
def compute_cross_rank_procedural_variance(
    rank_dirs: dict[int, Path],  # {4: Path(...), 8: Path(...), 16: Path(...)}
    dataset: str = "mengrowth",  # or "bratsmen_test"
) -> pd.DataFrame:
    """
    Compare procedural uncertainty magnitude across LoRA ranks.

    For each rank r:
        - Load the volume CSV
        - Compute per-scan ensemble std (or MAD_scaled)
        - Report summary statistics: median std, mean std, IQR

    The relationship between rank and procedural variance tests whether
    higher-rank adapters (more parameters, less regularization) produce
    greater inter-member diversity. By the Jiménez et al. framework (§B.2),
    regularization reduces variance but increases bias — so if procedural
    variance increases with rank while bias (measured on BraTS-MEN test)
    decreases, we observe the classical bias-variance trade-off in action.

    Returns DataFrame with columns:
        rank, median_logvol_std, mean_logvol_std, iqr_logvol_std,
        median_abs_bias (if GT available), mean_abs_bias (if GT available)
    """
```

#### 2.4.4 Function: `generate_taxonomy_table`

```python
def generate_taxonomy_table(
    bias_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    rank: int,
    n_members: int,
    n_mengrowth_patients: int,
    n_bratsmen_train: int,
) -> dict:
    """
    Generate the five-source epistemic uncertainty taxonomy summary
    for a single LoRA rank configuration.

    Returns a dictionary suitable for JSON serialization and thesis
    inclusion, with the following structure:

    {
        "config": {"rank": r, "n_members": M, "seed": 42},
        "taxonomy": {
            "approximation_error": {
                "status": "not_quantifiable",
                "note": "Requires oracle access to true p(y|x); BrainSegFounder is
                         a universal approximator so this term vanishes in theory
                         (Hornik et al., 1989) but not in practice with finite width."
            },
            "estimation_bias": {
                "status": "diagnosed",
                "median_abs_bias_logvol": ...,
                "mean_abs_bias_logvol": ...,
                "pct_scans_bias_gt_std": ...,  # % where |bias| > ensemble std
                "note": "Sources: LoRA low-rank constraint (rank=r),
                         frozen encoder bias, finite BraTS-MEN training set."
            },
            "procedural_uncertainty": {
                "status": "measured",
                "median_logvol_std": ...,
                "mean_logvol_std": ...,
                "note": "Ensemble volume std across M members with different
                         random seeds on the same dataset."
            },
            "data_uncertainty": {
                "status": "partially_captured",
                "note": "LOPO-CV varies the MenGrowth training set across folds,
                         capturing some data uncertainty at the growth-model level.
                         Not propagated per-fold as observation-level sigma_v,k.
                         BraTS-MEN training set (N=~800) is fixed across all members."
            },
            "distributional_uncertainty": {
                "status": "mitigated",
                "note": "BraTS-MEN (multi-site, standardized) → Andalusian cohort
                         (~10 scanners). ComBat harmonization applied. Residual shift
                         not quantified at segmentation level."
            }
        },
        "calibration": {
            "coverage_95": ...,
            "coverage_90": ...,
            "coverage_deficit_95": ...  # nominal - empirical
        },
        "recommendation": "Interpret sigma_v,k as a lower bound on total
                           segmentation-derived epistemic uncertainty."
    }
    """
```

#### 2.4.5 Function: `plot_bias_vs_std`

```python
def plot_bias_vs_std(
    bias_df: pd.DataFrame,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """
    Scatter plot of |bias| vs ensemble std per scan, analogous to
    Jiménez et al. (2026) Figure 8.

    Points above the diagonal (|bias| > std) indicate scans where the
    ensemble is confidently wrong — the bias contamination regime.

    Uses log-volume scale by default since downstream LME operates there.

    Produces: {output_path}/bias_vs_std_logvol_r{rank}.png
    """
```

**Plot specification:**
- x-axis: `logvol_ensemble_std` (procedural uncertainty)
- y-axis: `logvol_abs_bias` (estimation bias)
- Diagonal line: y = x (below = well-calibrated, above = bias-dominated)
- Color: by scanner ID if available, else uniform
- Annotation: fraction of points above diagonal
- Style: matplotlib, no seaborn dependency unless already in project

#### 2.4.6 Function: `plot_calibration_curve`

```python
def plot_calibration_curve(
    per_member_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    output_path: Path,
    n_levels: int = 20,
) -> None:
    """
    Plot nominal vs empirical coverage for the ensemble CI.

    Perfect calibration = diagonal. Below diagonal = overconfident
    (consistent with missing uncertainty components).

    Produces: {output_path}/calibration_coverage_r{rank}.png
    """
```

#### 2.4.7 Main orchestrator

```python
def run_epistemic_taxonomy_analysis(
    run_dir: Path,
    rank: int,
    n_members: int = 20,
    seed: int = 42,
) -> dict:
    """
    Full analysis pipeline for a single rank configuration.

    Steps:
        1. Load per_member_test_dice.csv and ensemble_test_dice.csv
        2. compute_volume_bias → bias_df
        3. compute_calibration_coverage → calibration_df
        4. generate_taxonomy_table → taxonomy_dict
        5. plot_bias_vs_std → PNG
        6. plot_calibration_curve → PNG
        7. Save taxonomy_dict → {run_dir}/evaluation/epistemic_taxonomy.json
        8. Save bias_df → {run_dir}/evaluation/bias_diagnostics.csv
        9. Log summary to stdout

    Returns: taxonomy_dict
    """
```

And a cross-rank wrapper:

```python
def run_cross_rank_comparison(
    base_output_dir: Path,
    ranks: list[int] = [4, 8, 16],
    n_members: int = 20,
    seed: int = 42,
) -> None:
    """
    Run taxonomy analysis for all ranks and produce comparison outputs.

    Steps:
        1. For each rank, call run_epistemic_taxonomy_analysis
        2. compute_cross_rank_procedural_variance → cross_rank_df
        3. Save cross_rank_df → {base_output_dir}/cross_rank_epistemic_summary.csv
        4. Save combined taxonomy → {base_output_dir}/cross_rank_taxonomy.json
        5. Plot: rank vs median procedural std (bar chart with error bars)
        6. Plot: rank vs median |bias| (bar chart) — tests bias-variance trade-off
    """
```

### 2.5 Output files

For each rank configuration `r{rank}_M20_s42/evaluation/`:

| File | Content |
|------|---------|
| `epistemic_taxonomy.json` | Full taxonomy dict (§2.4.4) |
| `bias_diagnostics.csv` | Per-scan bias, std, ratio |
| `bias_vs_std_logvol_r{rank}.png` | Scatter plot |
| `calibration_coverage_r{rank}.png` | Calibration curve |

At the top level `{base_output_dir}/`:

| File | Content |
|------|---------|
| `cross_rank_epistemic_summary.csv` | Rank × {median_std, median_bias, ...} |
| `cross_rank_taxonomy.json` | All ranks combined |
| `rank_vs_procedural_std.png` | Bar chart |
| `rank_vs_bias.png` | Bar chart |

### 2.6 Dependencies

All already in the project environment — no new packages:

- `pandas`, `numpy` — data manipulation
- `scipy.stats` — `t.ppf` for CI multipliers
- `matplotlib` — plotting
- `json`, `pathlib`, `logging` — standard library

### 2.7 Coding conventions

Follow the project's established patterns:

- **Type hints** on all function signatures.
- **Docstrings** on all public functions (Google style, matching existing code).
- **Structured logging** via `logging.getLogger(__name__)`.
- **No magic numbers** — all thresholds (e.g., `1e-8` epsilon) as module-level constants.
- **Dataclass** for configuration if more than 3 parameters are passed around.
- Use `pathlib.Path` consistently (not string paths).

### 2.8 Testing / verification criteria

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| ET-T1 | `bias_diagnostics.csv` has one row per test scan | `len(df) == n_test_scans` |
| ET-T2 | `logvol_bias_to_std_ratio` is finite everywhere | No NaN or Inf values |
| ET-T3 | Calibration coverage at 95% is reported | Value in [0, 1] |
| ET-T4 | Cross-rank CSV has exactly 3 rows (ranks 4, 8, 16) | `len(df) == 3` |
| ET-T5 | Taxonomy JSON is valid and contains all 5 sources | All keys present |
| ET-T6 | Bias-vs-std plot file exists and is non-empty | File size > 1 KB |
| ET-T7 | If coverage < 0.95 at nominal 95%, this is consistent with the Jiménez et al. prediction | Logged as expected |

---

## 3. Task C — Documentation update: thesis-level framing

### 3.1 Where to add

This is guidance for the thesis writing, not a code change. However, to keep it tracked, add a brief note to the top of `PLAN_OF_ACTION_v1.md` (or a new section at the end) titled:

**"Epistemic Uncertainty Framing (per Jiménez et al., 2026)"**

### 3.2 Content

```markdown
## Epistemic Uncertainty Framing (per Jiménez et al., 2026)

The LoRA ensemble's volume uncertainty ($\sigma_{v,k}$) must be explicitly framed in
the thesis as **procedural uncertainty** — a lower bound on total epistemic uncertainty.
This framing is supported by the formal analysis in Jiménez, Jürgens & Waegeman
(2026, arXiv:2505.23506v3), which demonstrates that Deep Ensembles (and structurally
equivalent methods like our LoRA ensemble) capture only one of five identified sources
of epistemic uncertainty.

### Thesis sections affected:

1. **Methods — Uncertainty propagation (§X.Y):** When describing the pipeline
   $p(\mathbf{s}|\mathbf{x}) \to p(V|\mathbf{s}) \to p(V(t^*)|\{V_k\})$, state explicitly:
   "The ensemble volume standard deviation $\sigma_{v,k}$ captures procedural uncertainty
   — sensitivity to random initialization and training order — but not data
   uncertainty (sensitivity to the training set) or estimation bias (systematic
   deviation from the Bayes-optimal segmenter). It therefore constitutes a lower
   bound on the total segmentation-derived epistemic uncertainty
   (Jiménez et al., 2026)."

2. **Results — Uncertainty diagnostics (§X.Y):** Report the empirical findings:
   - What fraction of test scans have |bias| > ensemble std?
   - What is the empirical coverage at 95% nominal?
   - How does procedural uncertainty change with LoRA rank?

3. **Discussion — Limitations (§X.Y):** The five-source taxonomy table (from
   `epistemic_taxonomy.json`) should appear as a thesis table, demonstrating
   methodological maturity and honest accounting. The key claim:
   "Our framework provides principled propagation of **procedural** uncertainty
   from segmentation through growth modeling, while acknowledging that estimation
   bias, data uncertainty, and distributional shift represent additional
   unquantified components. The LME's patient-level random effects and the
   Gompertz mean function constraint provide partial structural mitigation of
   estimation bias, consistent with the favorable bias-variance trade-off at
   our cohort size."

4. **Discussion — Why LME beats GP (§X.Y):** The Jiménez et al. framework
   provides the theoretical justification: regularization reduces variance at the
   cost of increased bias. At N=31–58, the variance reduction from LME's strong
   parametric constraints outweighs the bias increase — hence LME ($R^2_{\log}
   \approx 0.387$) outperforms ScalarGP ($\approx 0.077$) and HGP ($\approx -0.037$).
   This is the classical bias-variance trade-off operating exactly as theory predicts.
```

### 3.3 Verification

The agent should confirm that:
- The note is appended without disrupting existing content.
- Section references (§X.Y) are left as placeholders — Mario will fill them in during thesis writing.
- The cited R² values match the current project memory.

---

## 4. Execution order and dependencies

```
Task A (documentation)  ─── no code dependency ───→  can be done first
Task C (thesis framing) ─── no code dependency ───→  can be done in parallel with A

Task B (analysis code)  ─── depends on existing CSVs ───→  do after A
  └─ Step 1: Inspect actual CSV column names
  └─ Step 2: Implement epistemic_taxonomy.py
  └─ Step 3: Run on r=4, r=8, r=16
  └─ Step 4: Verify outputs (ET-T1 through ET-T7)
```

Estimated effort: ~2–3 hours for an agent with access to the codebase.

---

## 5. What this does NOT cover (out of scope)

- **Modifying the LME/GP code** to ingest the taxonomy outputs. The current $\sigma_{v,k}$ propagation mechanism is unchanged; we are adding diagnostics, not modifying the pipeline.
- **Implementing a reference distribution** (Jiménez et al., Definition 3.1). This would require retraining ensemble members on resampled BraTS-MEN subsets — computationally expensive (~20 × 20 × 6h = 2400 GPU-hours) and out of scope for the current thesis timeline.
- **Modifying the LoRA training procedure.** The ensemble is already trained.
- **Quantifying distributional uncertainty** (BraTS-MEN → Andalusian shift) at the segmentation level. This would require a held-out Andalusian test set with expert segmentation labels, which is not available.

---

## 6. Summary of deliverables

| # | Deliverable | Type | Location |
|---|------------|------|----------|
| 1 | Updated `UNCERTAINTY_SEGMENTATION_SUMMARY.md` with §8.1 + refs | Doc edit | Project knowledge |
| 2 | `epistemic_taxonomy.py` with 7 functions + orchestrator | New file | `experiments/uncertainty_segmentation/engine/` |
| 3 | Per-rank JSON + CSV + 2 plots | Analysis output | `{run_dir}/evaluation/` |
| 4 | Cross-rank comparison CSV + JSON + 2 plots | Analysis output | `{base_output_dir}/` |
| 5 | Thesis framing note appended to `PLAN_OF_ACTION_v1.md` | Doc edit | Project knowledge |
